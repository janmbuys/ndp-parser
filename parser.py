# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source

import argparse
import os
import sys
import math
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

import rnn_lm
import rnn_encoder
import utils


def get_sentence_batch(source, evaluation=False):
    data = Variable(source.word_tensor[:-1], volatile=evaluation)
    target = Variable(source.word_tensor[1:].view(-1))
    return data, target


def create_length_histogram(sentences):
  token_count = 0
  missing_token_count = 0

  sent_length = defaultdict(int)
  for sent in sentences:
    sent_length[len(sent)] += 1
    token_count += len(sent)
    missing_token_count += min(len(sent), 50)
  lengths = list(sent_length.keys())
  lengths.sort()
  print('Token count %d. length 50 count %d prop %.4f.'
        % (token_count, missing_token_count,
            missing_token_count/token_count))

  cum_count = 0
  with open(working_path + 'histogram', 'w') as fh:
    for length in lengths:
      cum_count += sent_length[length]
      fh.write((str(length) + '\t' + str(sent_length[length]) + '\t' 
                + str(cum_count) + '\n'))
  print('Created histogram')   


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True,
                      help='Directory of Annotated CONLL files')
  parser.add_argument('--working_dir', required=True,
                      help='Working directory for parser')
  parser.add_argument('--train_name',
                      help='Train file name (excluding .conll)',
                      default='train')
  parser.add_argument('--dev_name',
                      help='Dev file name (excluding .conll)', 
                      default='dev')
  parser.add_argument('--test_name',
                      help='Test file name (excluding .conll)', 
                      default='test')
  parser.add_argument('--test_file',
                      help='Raw text file to parse with trained model', 
                      metavar='FILE')
  parser.add_argument('--replicate_rnng_data', action='store_true', 
                      default=False)
  parser.add_argument('--reset_vocab', action='store_true', 
                      default=False)

  parser.add_argument('--embedding_size', type=int, default=200,
                      help='size of word embeddings')
  parser.add_argument('--hidden_size', type=int, default=200, 
                      help='humber of hidden units per layer')
  parser.add_argument('--num_layers', type=int, default=2, # make 1
                      help='number of layers')
  parser.add_argument('--lr', type=float, default=20, #TODO check
                      help='initial learning rate')
  parser.add_argument('--grad_clip', type=float, default=0.5, #TODO check
                      help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=10,
                      help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                      help='batch size') # original default 20
  parser.add_argument('--bptt', type=int, default=20, # not using now
                      help='sequence length')
  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  parser.add_argument('--cuda', action='store_true',
                      help='use CUDA')
  parser.add_argument('--small_data', action='store_true',
                      help='use small version of dataset')
  parser.add_argument('--logging_interval', type=int, default=5000, 
                      metavar='N', help='report ppl every x steps')
  parser.add_argument('--save_model', type=str,  default='model.pt',
                      help='path to save the final model')

  args = parser.parse_args()

  torch.manual_seed(args.seed)
  if torch.cuda.is_available() and args.cuda:
    torch.cuda.manual_seed(args.seed)

  working_path = args.working_dir + '/'
  data_path = args.data_dir + '/' 

  # Prepare training data

  vocab_path = Path(working_path + 'vocab')
  if vocab_path.is_file() and not args.reset_vocab:
    sentences, word_vocab, pos_vocab, rel_vocab = utils.read_sentences_given_vocab(
        data_path, args.train_name, working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data)
  else:     
    print('Preparing vocab')
    sentences, word_vocab, pos_vocab, rel_vocab = utils.read_sentences_create_vocab(
        data_path, args.train_name, working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data)

  # Read dev and test files with given vocab
  dev_sentences, _, _, _ = utils.read_sentences_given_vocab(
        data_path, args.dev_name, working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data)

  test_sentences, _,  _, _ = utils.read_sentences_given_vocab(
        data_path, args.test_name, working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data)

  if args.small_data:
    sentences = sentences[:500]
    dev_sentences = dev_sentences[:100]

  create_length_histogram(sentences)

  # Extract oracle sentences
  for sent in sentences[:5]:
    actions, labels = utils.oracle(sent.conll)

  def train_lm():
    lr = args.lr
    vocab_size = len(word_vocab)
    batch_size = 1

    # Build the model
    model = rnn_lm.RNNLM(vocab_size, args.embedding_size,
      args.hidden_size, args.num_layers)
    #if args.cuda: #TODO 
    #  model.cuda()

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr) #TODO use

    # Loop over epochs.
    # Sentence iid, batch size 1 training.
    prev_val_loss = None
    for epoch in range(1, args.epochs+1):
      epoch_start_time = time.time()

      total_loss = 0
      for i, train_sent in enumerate(sentences):
        # Training loop
        start_time = time.time()
        data, targets = get_sentence_batch(train_sent)
        model.zero_grad()
        hidden_state = model.init_hidden(batch_size)
        output, hidden_state = model(data, hidden_state)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()

        utils.clip_grad_norm(model.parameters(), args.grad_clip)
        #optimizer.step() 
        for p in model.parameters():
          p.data.add_(-lr, p.grad.data)
        total_loss += loss.data

        if i % args.logging_interval == 0 and i > 0:
          cur_loss = total_loss[0] / args.logging_interval
          elapsed = time.time() - start_time
          print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
              epoch, i, len(sentences), lr,
              elapsed * 1000 / args.logging_interval, cur_loss, 
              math.exp(cur_loss)))
          total_loss = 0
          start_time = time.time()

      # Evualate
      val_batch_size = 1
      total_loss = 0
      total_length = 0
      for val_sent in dev_sentences:
        hidden_state = model.init_hidden(val_batch_size)
        data, targets = get_sentence_batch(val_sent, evaluation=True)
        output, hidden_state = model(data, hidden_state)
        output_flat = output.view(-1, vocab_size)
        total_loss += criterion(output_flat, targets).data
        total_length += len(data)
      val_loss = total_loss[0] / total_length

      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
      print('-' * 89)
      # Anneal the learning rate.
      if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4
      prev_val_loss = val_loss

      # save the model
      if args.save_model != '':
        model_fn = working_path + args.save_model
        with open(model_fn, 'wb') as f:
          torch.save(model, f)

  train_lm()
