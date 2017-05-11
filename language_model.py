# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source

import argparse
import math
import os
import random
import sys
import time

from collections import defaultdict
from pathlib import Path

import numpy as py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import rnn_lm
import data_utils
import nn_utils

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True,
                      help='Directory of Annotated CONLL files')
  parser.add_argument('--data_working_dir', required=True,
                      help='Working directory for data files')
  parser.add_argument('--working_dir', required=True,
                      help='Working directory for output')
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
  parser.add_argument('--txt_data_fixed_vocab', action='store_true', 
                      default=False)

  parser.add_argument('--embedding_size', type=int, default=128,
                      help='size of word embeddings')
  parser.add_argument('--hidden_size', type=int, default=128, 
                      help='humber of hidden units per layer')
  parser.add_argument('--num_layers', type=int, default=1, # original 2
                      help='number of layers')
  parser.add_argument('--lr', type=float, default=0.0001, # default 20
                      help='initial learning rate')
  parser.add_argument('--grad_clip', type=float, default=5, # default 0.5 check
                      help='gradient clipping')
  parser.add_argument('--num_init_lr_epochs', type=int, default=-1, 
                      help='number of epochs before learning rate decay')
  parser.add_argument('--patience', type=int, default=10, 
                      help='Stop training if not improving for some number of epochs')
  parser.add_argument('--lr_decay', type=float, default=1.1,
                      help='learning rate decay per epoch')
  parser.add_argument('--tie_weights', action='store_true',
                      help='tie the word embedding and softmax weights')
  parser.add_argument('--reduce_lr', action='store_true',
                      help='reduce lr if val ppl does not improve')
  parser.add_argument('--xavier_init', action='store_true',
                      help='Xavier initialization')
  parser.add_argument('--init_weight_range', type=float, default=0.1,
                      help='weight initialization range')

  parser.add_argument('--dropout', type=float, default=0.0, 
                      help='dropout rate')
  parser.add_argument('--epochs', type=int, default=100,
                      help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                      help='batch size') 
  parser.add_argument('--bptt', type=int, default=20, # not using now
                      help='sequence length')
  parser.add_argument('--adam', action='store_true',
                      help='use Adam optimizer')

  parser.add_argument('--cuda', action='store_true',
                      help='use CUDA')
  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  parser.add_argument('--small_data', action='store_true',
                      help='use small version of dataset')
  parser.add_argument('--logging_interval', type=int, default=5000, 
                      metavar='N', help='report ppl every x steps')
  parser.add_argument('--save_model', type=str,  default='model.pt',
                      help='path to save the final model')

  args = parser.parse_args()

  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    assert torch.cuda.is_available(), 'Cuda not available.'
    torch.cuda.manual_seed(args.seed)

  data_path = args.data_dir + '/' 
  data_working_path = args.data_working_dir + '/'
  working_path = args.working_dir + '/'

  # Prepare training data
  vocab_path = Path(data_working_path + 'vocab')
  if args.txt_data_fixed_vocab: # assume that way vocab is constructed won't change
    sentences, word_vocab = data_utils.read_sentences_txt_fixed_vocab(
        data_path, args.train_name, data_working_path)
  elif vocab_path.is_file() and not args.reset_vocab:
    sentences, word_vocab, _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data)
  else:     
    print('Preparing vocab')
    sentences, word_vocab, _, _ = data_utils.read_sentences_create_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data)

  # Read dev and test files with given vocab
  if args.txt_data_fixed_vocab:
    dev_sentences, _ = data_utils.read_sentences_txt_given_fixed_vocab(
        data_path, args.dev_name, data_working_path)
    test_sentences, _ = data_utils.read_sentences_txt_given_fixed_vocab(
        data_path, args.test_name, data_working_path)
  else:
    dev_sentences, _, _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.dev_name, data_working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data)

    test_sentences, _,  _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.test_name, data_working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data)

  if args.small_data:
    sentences = sentences[:1000]
    dev_sentences = dev_sentences #[:100]

  #data_utils.create_length_histogram(sentences)

  def train():
    lr = args.lr
    vocab_size = len(word_vocab)
    batch_size = args.batch_size

    # Build the model
    model = rnn_lm.RNNLM(vocab_size, args.embedding_size,
      args.hidden_size, args.num_layers, args.dropout, 
      args.init_weight_range, args.xavier_init, args.tie_weights, args.cuda)
    if args.cuda:
      model.cuda()

    criterion = nn.CrossEntropyLoss(size_average=False)
    if args.adam:
      optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
      optimizer = optim.SGD(model.parameters(), lr=lr)

    # Loop over epochs.
    # Sentence iid, batch size 1 training.
    prev_val_loss = None
    patience_count = 0
    for epoch in range(1, args.epochs+1):
      epoch_start_time = time.time()
      random.shuffle(sentences)
      sentences.sort(key=len) 
      model.train()

      total_loss = 0
      total_num_tokens = 0
      global_loss = 0
      global_num_tokens = 0
      batch_count = 0

      start_time = time.time()
      i = 0  
      while i < len(sentences):
        # Training loop
        length = len(sentences[i])
        j = i + 1
        while (j < len(sentences) and len(sentences[j]) == length
               and (j - i) < batch_size):
          j += 1
        # dimensions [length x batch]
        data, targets = nn_utils.get_sentence_batch(
            [sentences[k] for k in range(i, j)], args.cuda)
        local_batch_size = j - i
        i = j
        batch_count += 1

        model.zero_grad()
        hidden_state = model.init_hidden(local_batch_size)
        output, hidden_state = model(data, hidden_state)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()

        if args.grad_clip > 0:
          nn_utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step() 
        total_loss += loss.data
        global_loss += loss.data

        batch_tokens = (data.size()[0] - 1)*local_batch_size
        total_num_tokens += batch_tokens
        global_num_tokens += batch_tokens

        if batch_count % args.logging_interval == 0 and i > 0:
          cur_loss = total_loss[0] / total_num_tokens
          elapsed = time.time() - start_time
          print('| epoch {:3d} | {:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
              epoch, batch_count, lr,
              elapsed * 1000 / args.logging_interval, cur_loss, 
              math.exp(cur_loss)))
          total_loss = 0
          total_num_tokens = 0
          start_time = time.time()
        
      avg_global_loss = global_loss[0] / global_num_tokens
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | {:5d} batches | tokens {:5d} | loss {:5.2f} | ppl {:8.2f}'.format(
          epoch, (time.time() - epoch_start_time), batch_count, global_num_tokens,
          avg_global_loss, math.exp(avg_global_loss)))

      # Evaluate
      val_batch_size = 1
      total_loss = 0
      total_loss_direct = 0
      total_length = 0
      total_length_more = 0
      model.eval()
      log_normalize = nn.LogSoftmax()
      calculate_direct = False

      for val_sent in dev_sentences:
        hidden_state = model.init_hidden(val_batch_size)
        data, targets = nn_utils.get_sentence_batch([val_sent], args.cuda, evaluation=True)
        output, hidden_state = model(data, hidden_state)
        output_flat = output.view(-1, vocab_size)
        total_loss += criterion(output_flat, targets).data
        assert output_flat.size()[0] == len(val_sent)
        # direct loss calculation
        if calculate_direct:
          word_ids = [int(x) for x in targets.view(-1).data]
          word_dist_list = log_normalize(output_flat)
          for i, word_id in enumerate(word_ids):
            total_loss_direct -= nn_utils.to_numpy(word_dist_list[i, word_id])
        total_length += len(val_sent) - 1 
        total_length_more += len(val_sent) 

      val_loss = total_loss[0] / total_length
      val_loss_more = total_loss[0] / total_length_more
      if calculate_direct:
        val_loss_direct = total_loss_direct[0] / total_length

      print('| end of epoch {:3d} | time: {:5.2f}s | {:5d} tokens | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         total_length, val_loss, math.exp(val_loss)))
      print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(val_loss_more,
          math.exp(val_loss_more)))
      if calculate_direct:
        print('                   | valid loss direct {:5.2f} | valid ppl {:8.2f}'.format(val_loss_direct, math.exp(val_loss_direct)))

      print('-' * 89)
      # Anneal the learning rate.
      if (not args.adam and args.num_init_lr_epochs > 0 
          and epoch >= args.num_init_lr_epochs):
        if args.reduce_lr and val_loss > prev_val_loss:
          lr /= 2
        else:
          lr = lr / args.lr_decay
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
      if prev_val_loss and val_loss > prev_val_loss:
        patience_count += 1
      else:
        patience_count = 0
        prev_val_loss = val_loss
        # save the model
        if args.save_model != '':
          model_fn = working_path + args.save_model
          with open(model_fn, 'wb') as f:
            torch.save(model, f)
      if args.patience > 0 and patience_count >= args.patience:
        break

  train()
