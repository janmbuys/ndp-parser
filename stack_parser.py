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
  lr = args.lr # TODO add new LM lr adjustments etc
 
  prev_val_loss = None
  for epoch in range(1, args.epochs+1):
    print('Start unsup training epoch %d' % epoch)
    epoch_start_time = time.time()
    
    random.shuffle(sentences)
    sentences.sort(key=len)
    stack_model.train()

    total_loss = 0 
    global_loss = 0 
    total_num_tokens = 0 
    global_num_tokens = 0 
    batch_count = 0

    start_time = time.time()
    print('Training size {:3d}'.format(len(sentences)))

    i = 0  
    while i < len(sentences):
      # Training loop
      length = len(sentences[i])
      j = i + 1
      while (j < len(sentences) and len(sentences[j]) == length
             and (j - i) < batch_size):
        j += 1
      # dimensions [length x batch]
      sentence_data = nn_utils.get_sentence_data_batch(
          [sentences[k] for k in range(i, j)], args.cuda)
      local_batch_size = j - i
      i = j
      batch_count += 1

      stack_model.zero_grad()
      loss = stack_model.neg_log_likelihood(sentence_data)
      #print(loss)

      if loss is not None:
        loss.backward()
        if args.grad_clip > 0:
          nn_utils.clip_grad_norm(params, args.grad_clip)
        optimizer.step() 
        total_loss += loss.data
        global_loss += loss.data

      batch_tokens = (sentence_data.size()[0] - 1)*local_batch_size
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

    # Eval dev set ppl  
    val_batch_size = 1
    total_loss = 0
    total_length = 0
    dev_losses = []
    decode_start_time = time.time()

    stack_model.eval()
    normalize = nn.Sigmoid() 

    for val_sent in dev_sentences:
      sentence_data = nn_utils.get_sentence_data_batch([val_sent], args.cuda,
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
      sentence_data = nn_utils.get_sentence_data_batch([val_sent], args.cuda,
          evaluation=True)
      transition_logits, actions, shift_dependents, reduce_dependents = stack_model.forward(sentence_data)
      #action_str =  ' '.join(['SH' if act == 0 else 'RE' for act in actions])
      #print(action_str)
      #print('shift_dependents') 
      #print(shift_dependents) 
      #print('reduce_dependents') 
      #print(reduce_dependents) 
      #TODO print output, easiest is to define class in data_utils

    val_loss = 0
    print('| decode time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
         (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

    if args.save_model != '':
      model_fn = working_path + args.save_model + '_stack.pt'
      with open(model_fn, 'wb') as f:
        torch.save(stack_model, f)

