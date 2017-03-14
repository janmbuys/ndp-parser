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
import rnn_encoder
import classifier
import utils

def filter_logits(logits, targets):
  logits_filtered = [logit for logit in logits if logit is not None]
  targets_filtered = [target for (logit, target) in zip(logits, targets)
                          if logit is not None]
  if logits_filtered:
    if args.cuda:
      #print(logits_filtered)
      #print(targets_filtered)
      target_var = Variable(torch.LongTensor(targets_filtered)).cuda()
    else:
      target_var = Variable(torch.LongTensor(targets_filtered))

    output = torch.cat(logits_filtered, 0)
    return output, target_var
  else:
    return None, None


# Training oracle for single example
def train_oracle(conll, encoder_features, transition_model, relation_model=None):
  stack = utils.ParseForest([])
  buf = utils.ParseForest([conll[0]])
  buffer_index = 0
  sent_length = len(conll)

  num_children = [0 for _ in conll]
  for token in conll:
    num_children[token.parent_id] += 1

  actions = []
  transition_logits = []
  relation_logits = []
  labels = []

  while buffer_index < sent_length or len(stack) > 1:
    transition_logit = None
    relation_logit = None
    action = utils._SH
    if buffer_index == sent_length:
      action = utils._RA
    elif len(stack) > 1: # allowed to ra or la
      s0 = stack.roots[-1].id 
      s1 = stack.roots[-2].id 
      b = buf.roots[0].id 
      if stack.roots[-1].parent_id == b: # candidate la
        if len(stack.roots[-1].children) == num_children[s0]:
          action = utils._LA 
      elif stack.roots[-1].parent_id == s1: # candidate ra
        if len(stack.roots[-1].children) == num_children[s0]:
          action = utils._RA 
      if args.cuda:
        feature_positions = Variable(torch.LongTensor([s0, s1, b])).cuda()
      else:
        feature_positions = Variable(torch.LongTensor([s0, s1, b]))

      features = torch.index_select(encoder_features, 0, 
                                    feature_positions)
      transition_logit = transition_model(features)      
      if relation_model is not None and action != utils._SH:
        relation_logit = relation_model(features)      

    actions.append(action)
    transition_logits.append(transition_logit)
    relation_logits.append(relation_logit)
     
    label = -1 if action == utils._SH else stack.roots[-1].relation_id
    labels.append(label)

    # excecute action
    if action == utils._SH:
      stack.roots.append(buf.roots[0]) 
      buffer_index += 1
      if buffer_index == sent_length:
        buf = utils.ParseForest([])
      else:
        buf = utils.ParseForest([conll[buffer_index]])
    else:  
      assert len(stack) > 0
      child = stack.roots.pop()
      if action == utils._LA:
        buf.roots[0].children.append(child) 
      else:
        stack.roots[-1].children.append(child)
  return transition_logits, actions, relation_logits, labels


def greedy_decode(conll, encoder_features, transition_model, relation_model=None):
  stack = utils.ParseForest([])
  buf = utils.ParseForest([conll[0]])
  buffer_index = 0
  sent_length = len(conll)

  actions = []
  labels = []
  transition_logits = []
  relation_logits = []

  while buffer_index < sent_length or len(stack) > 1:
    transition_logit = None
    relation_logit = None
    action = utils._SH
    label = ''
    if buffer_index == sent_length:
      action = utils._RA
    elif len(stack) > 1: # allowed to ra or la
      s0 = stack.roots[-1].id 
      s1 = stack.roots[-2].id 
      b = buf.roots[0].id 
      
      if args.cuda:
        feature_positions = Variable(torch.LongTensor([s0, s1, b])).cuda()
      else:
        feature_positions = Variable(torch.LongTensor([s0, s1, b]))

      features = torch.index_select(encoder_features, 0, 
                                    feature_positions)
      #TODO rather score transition and relation jointly for greedy choice
      transition_logit = transition_model(features)
      transition_logit_np = transition_logit.type(torch.FloatTensor).data.numpy()
      action = int(transition_logit_np.argmax(axis=1)[0])
      if relation_model is not None and action != utils._SH:
        relation_logit = relation_model(features)      
        relation_logit_np = relation_logit.type(torch.FloatTensor).data.numpy()

        label = int(relation_logit_np.argmax(axis=1)[0])
      
    actions.append(action)
    transition_logits.append(transition_logit)
    if relation_model is not None:
      relation_logits.append(relation_logit)
      labels.append(label)

    # excecute action
    if action == utils._SH:
      stack.roots.append(buf.roots[0]) 
      buffer_index += 1
      if buffer_index == sent_length:
        buf = utils.ParseForest([])
      else:
        buf = utils.ParseForest([conll[buffer_index]])
    else:  
      assert len(stack) > 0
      child = stack.roots.pop()
      if action == utils._LA:
        buf.roots[0].children.append(child) 
        conll[child.id].pred_parent_id = buf.roots[0].id
      else:
        stack.roots[-1].children.append(child)
        conll[child.id].pred_parent_id = stack.roots[-1].id
      if relation_model is not None:
        conll[child.id].pred_relation_id = label
  return conll, transition_logits, actions, relation_logits, labels
 

def get_sentence_batch(source, use_cuda, evaluation=False):
  #TODO use target only for language modelling...
  if use_cuda:
    data = Variable(source.word_tensor[:-1], volatile=evaluation).cuda()
    target = Variable(source.word_tensor[1:].view(-1)).cuda()
  else:
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

  parser.add_argument('--embedding_size', type=int, default=128,
                      help='size of word embeddings')
  parser.add_argument('--hidden_size', type=int, default=128, 
                      help='humber of hidden units per layer')
  parser.add_argument('--num_layers', type=int, default=1, # original 2
                      help='number of layers')
  parser.add_argument('--lr', type=float, default=0.01, #TODO default 20 check
                      help='initial learning rate')
  parser.add_argument('--grad_clip', type=float, default=5, #TODO default 0.5 check
                      help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=100,
                      help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                      help='batch size') # original default 20
  parser.add_argument('--bptt', type=int, default=20, # not using now
                      help='sequence length')

  parser.add_argument('--relation_prediction', dest='predict_relations',
                      action='store_true', help='predict relation labels.')
  parser.add_argument('--no_relation_prediction', dest='predict_relations',
                      action='store_false', 
                      help='do not predict relation labels.')
  parser.set_defaults(predict_relations=True)

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

  #create_length_histogram(sentences)

  # Extract oracle sentences
  #for sent in sentences[:5]:
  #  actions, labels = utils.oracle(sent.conll)

  def train():
    vocab_size = len(word_vocab)
    num_relations = len(rel_vocab)
    num_transitions = 3
    num_features = 3

    batch_size = args.batch_size

    # Build the model
    encoder_model = rnn_encoder.RNNEncoder(vocab_size, 
        args.embedding_size, args.hidden_size, args.num_layers, args.cuda)

    feature_size = args.hidden_size # x2 for bidirectional
    transition_model = classifier.Classifier(num_features, feature_size, 
        args.hidden_size, num_transitions, args.cuda) 
    if args.predict_relations:
      relation_model = classifier.Classifier(num_features, feature_size, 
          args.hidden_size, num_relations, args.cuda)
    else:
      relation_model = None

    if args.cuda:
      encoder_model.cuda()
      transition_model.cuda()
      if args.predict_relations:
        relation_model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.predict_relations:
      params = (list(encoder_model.parameters()) +
                list(transition_model.parameters()) +
                list(relation_model.parameters()))
    else:
      params = (list(encoder_model.parameters()) +
                list(transition_model.parameters()))

    optimizer = optim.Adam(params, lr=args.lr)
   
    prev_val_loss = None
    for epoch in range(1, args.epochs+1):
      print('Start training epoch %d' % epoch)
      epoch_start_time = time.time()
      
      random.shuffle(sentences)
      sentences.sort(key=len) 
      #TODO batch for RNN encoder

      total_loss = 0 

      for i, train_sent in enumerate(sentences):
        # Training loop
        start_time = time.time()

        # sentence encoder
        sentence_data, _ = get_sentence_batch(train_sent, args.cuda)
        encoder_model.zero_grad()
        transition_model.zero_grad()
        if args.predict_relations:
          relation_model.zero_grad()
        encoder_state = encoder_model.init_hidden(batch_size)
        encoder_output = encoder_model(sentence_data, encoder_state)
 
        transition_logits, actions, relation_logits, labels = train_oracle(train_sent.conll, 
                encoder_output, transition_model, relation_model)

        # Filter out Nones to get training examples, then concatenate
        transition_output, action_var = filter_logits(transition_logits,
            actions)

        if transition_output is not None:
          loss = criterion(transition_output.view(-1, num_transitions),
                           action_var)
          if args.predict_relations:
            relation_output, label_var = filter_logits(relation_logits, labels)
            if relation_output is not None:
              loss += criterion(relation_output.view(-1, num_relations), label_var)

          loss.backward()
          utils.clip_grad_norm(params, args.grad_clip)
          optimizer.step() 
          total_loss += loss.data
 
        if i % args.logging_interval == 0 and i > 0:
          cur_loss = total_loss[0] / args.logging_interval
          elapsed = time.time() - start_time
          print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(  # | ppl {:8.2f}
              epoch, i, len(sentences), 
              elapsed * 1000 / args.logging_interval, cur_loss))
              #math.exp(cur_loss)))
          total_loss = 0
          start_time = time.time()

      val_batch_size = 1
      total_loss = 0
      total_length = 0
      conll_predicted = []

      for val_sent in dev_sentences:
        sentence_data, _ = get_sentence_batch(val_sent, args.cuda, evaluation=True)
        encoder_state = encoder_model.init_hidden(val_batch_size)
        encoder_output = encoder_model(sentence_data, encoder_state)
     
        predict, transition_logits, actions, relation_logits, labels = greedy_decode(
            val_sent.conll, encoder_output, transition_model, relation_model)
        conll_predicted.append(predict) 

        # Filter out Nones, then concatenate
        transition_logits_filtered = [logit for logit in transition_logits
                                      if logit is not None]
        actions_filtered = [act for (logit, act) in zip(transition_logits, 
                              actions) if logit is not None]
        if args.cuda:
          action_var = Variable(torch.LongTensor(actions_filtered)).cuda()
        else:
          action_var = Variable(torch.LongTensor(actions_filtered))

        if transition_logits_filtered != []:
          transition_output = torch.cat(transition_logits_filtered, 0)
          #total_loss += criterion(transition_output.view(-1, num_transitions),
          #                        action_var).data
          total_length += 1

      utils.write_conll(working_path + args.dev_name + '.' + str(epoch) + '.output.conll', conll_predicted)
      val_loss = 0
      #val_loss = total_loss[0] / total_length
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(
          epoch, (time.time() - epoch_start_time), val_loss))
            #'valid ppl {:8.2f}'
      print('-' * 89)

      # save the model
      if args.save_model != '':
        model_fn = working_path + args.save_model + '_encoder.pt'
        with open(model_fn, 'wb') as f:
          torch.save(encoder_model, f)
        model_fn = working_path + args.save_model + '_transition.pt'
        with open(model_fn, 'wb') as f:
          torch.save(transition_model, f)

  def train_lm():
    lr = args.lr
    vocab_size = len(word_vocab)
    batch_size = 1

    # Build the model
    model = rnn_lm.RNNLM(vocab_size, args.embedding_size,
      args.hidden_size, args.num_layers, args.cuda)
    if args.cuda:
      model.cuda()

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
        data, targets = get_sentence_batch(train_sent, args.cuda)
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
        data, targets = get_sentence_batch(val_sent, args.cuda, evaluation=True)
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

  train()
