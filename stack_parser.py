# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source; pytorch CRF example 

import argparse
import math
import os
import random
import sys
import time

from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import rnn_encoder
import classifier
import binary_classifier
import dp_stack
import data_utils
import nn_utils

#TODO now doing this inside dp_stack
def inside_score_compute(conll, encoder_features, transition_model, word_model):
  '''Compute inside score for training.'''
  assert word_model is not None
  sent_length = len(conll) # includes root, but not EOS
  log_normalize = nn.LogSoftmax()
  binary_normalize = nn.Sigmoid()

  # compute all sh/re and word probabilities
  seq_length = sent_length + 1

  # do simple arithmetic to access entries in the inside op
  def get_feature_index(i, j):
    return int((2*seq_length-i-1)*(i/2) + j-i-1)

  # batch feature computation
  features = nn_utils.batch_feature_selection(encoder_features, seq_length, args.cuda)
  re_probs_list = binary_normalize(transition_model(features))
  word_distr_list = log_normalize(word_model(features))

  # Get probabilities in python lists (entries in torch) - may not be sufficient
  #reduce_probs = []
  #word_log_probs = []
  #for i in range(seq_length-1):
  #  reduce_probs.append([None for _ in range(i+1)])
  #  word_log_probs.append([None for _ in range(i+1)])
  #  for j in range(i+1, seq_length):
  #    reduce_probs[-1].append(re_probs_list[counter, 0])
  #    if j < sent_length:
  #      if j < sent_length - 1:
  #        word_id = conll[j+1].word_id 
  #      else:
  #        word_id = data_utils._EOS
  #      word_log_probs[-1].append(word_dist[counter, word_id])
  #    counter += 1

  table_size = sent_length + 1
  table = []
  for _ in range(table_size):
    in_table = []
    for _ in range(table_size):
      in_table.append([None for _ in range(table_size)])
    table.append(in_table)

  # first word prob 
  init_features = nn_utils.select_features(encoder_features, [0, 0], args.cuda)
  init_word_dist = log_normalize(word_model(init_features).view(-1))
  table[0][0][1] = init_word_dist[conll[1].word_id]
  #print("computed features")

  for j in range(2, sent_length+1):
    #print(j)
    if j < sent_length - 1:
      word_id = conll[j+1].word_id 
    else:
      word_id = data_utils._EOS
    for i in range(j-1):
      index = get_feature_index(i, j-1)
      table[i][j-1][j] = (torch.log1p(-re_probs_list[index, 0]) 
                          + word_distr_list[index, word_id])
    for i in range(j-2, -1, -1):
      for l in range(max(i, 1)):
        block_scores = []
        score = None
        for k in range(i+1, j):
          re_score = torch.log(re_probs_list[get_feature_index(k, j), 0])
          t_score = table[l][i][k] + table[i][k][j] + re_score
          #score = score + t_score if score is not None else t_score
          block_scores.append(t_score)
        #table[l][i][j] = score
        table[l][i][j] = nn_utils.log_sum_exp(torch.cat(block_scores).view(1, -1))
  return table[0][0][sent_length]


def train_unsup(args, sentences, dev_sentences, test_sentences, word_vocab):
  vocab_size = len(word_vocab)
  num_features = 2
  batch_size = args.batch_size
  assert args.decompose_actions and args.generative and not args.bidirectional
  
  # Build the model
  feature_size = args.hidden_size
  stack_model = dp_stack.DPStack(vocab_size, args.embedding_size,
          args.hidden_size, args.num_layers, args.dropout, num_features,
          args.cuda)

  if args.cuda:
    stack_model.cuda()

  params = list(stack_model.parameters()) 
  optimizer = optim.Adam(params, lr=args.lr)
 
  prev_val_loss = None
  for epoch in range(1, args.epochs+1):
    print('Start unsup training epoch %d' % epoch)
    epoch_start_time = time.time()
    
    random.shuffle(sentences)
    #sentences.sort(key=len) # temp

    total_loss = 0 
    global_loss = 0 
    total_num_tokens = 0 
    global_num_tokens = 0 
    stack_model.train()

    start_time = time.time()
    print('Training size {:3d}'.format(len(sentences)))
    for i, train_sent in enumerate(sentences):
      # Training loop
      total_num_tokens += len(train_sent) - 1 
      global_num_tokens += len(train_sent) - 1 

      stack_model.zero_grad()
      normalize = nn.Sigmoid()

      sentence_data = nn_utils.get_sentence_batch(train_sent, args.cuda)
      
      loss = stack_model.neg_log_likelihood(sentence_data)

      if loss is not None:
        loss.backward()
        if args.grad_clip > 0:
          nn_utils.clip_grad_norm(params, args.grad_clip)
        optimizer.step() 
        total_loss += loss.data
        global_loss += loss.data

      if i % args.logging_interval == 0 and i > 0:
        cur_loss = total_loss[0] / total_num_tokens
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} '.format(   
            epoch, i, len(sentences), 
            elapsed * 1000 / args.logging_interval, cur_loss,
            math.exp(cur_loss)))
        total_loss = 0
        total_num_tokens = 0
        start_time = time.time()
 
    avg_global_loss = global_loss[0] / global_num_tokens
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | tokens {:5d} | loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), global_num_tokens,
        avg_global_loss, math.exp(avg_global_loss)))

    # Eval dev set ppl  
    val_batch_size = 1
    total_loss = 0
    total_length = 0
    dev_losses = []
    decode_start_time = time.time()

    stack_model.eval()
    normalize = nn.Sigmoid() 

    for val_sent in dev_sentences:
      sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda,
          evaluation=True)
      loss = stack_model.neg_log_likelihood(sentence_data)
      total_loss += loss
      dev_losses.append(loss.data[0])
      total_length += len(val_sent) - 1 

    val_loss = total_loss.data[0]  / total_length
    print('-' * 89)
    print('| eval time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
         (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

    stack_model.eval()
    decode_start_time = time.time()
    # Decode dev set 
    #TODO ppl from best path
    for val_sent in dev_sentences:
      sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda,
          evaluation=True)
      transition_logits, actions, shift_dependents, reduce_dependents = stack_model.forward(sentence_data)
      action_str =  ' '.join(['SH' if act == 0 else 'RE' for act in actions])
      print(action_str)
      print('shift_dependents') 
      print(shift_dependents) 
      print('reduce_dependents') 
      print(reduce_dependents) 
      #TODO print output, easiest is to define class in data_utils
    

    val_loss = 0
    print('| decode time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
         (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

    if args.save_model != '':
      model_fn = working_path + args.save_model + '_stack.pt'
      with open(model_fn, 'wb') as f:
        torch.save(stack_model, f)

