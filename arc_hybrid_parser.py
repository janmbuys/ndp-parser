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

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import rnn_encoder
import classifier
import binary_classifier
import data_utils
import nn_utils

def decompose_transitions(actions):
  stackops = []
  directions = []
  
  for action in actions:
    if action == data_utils._SH:
      stackops.append(data_utils._SSH)
      directions.append(None)
    else:
      stackops.append(data_utils._SRE)
      if action == data_utils._LA:
        directions.append(data_utils._DLA)
      else:  
        directions.append(data_utils._DRA)
  
  return stackops, directions



def backtrack_path(table, split_indexes, directions, l, i, j):
   """ Find action sequence for best path. """
   #TODO add option for only sh/re
   #TODO add label prediction
   if i == j - 1:
     return [data_utils._SH]
   else:
     k = split_indexes[l, i, j]
     direct = directions[l, i, j]
     act = data_utils._LA if direct == data_utils._DLA else data_utils._RA
     return (backtrack_path(table, split_indexes, directions, l, i, k)
             + backtrack_path(table, split_indexes, directions, i, k, j) 
             + [act])


def decode_action_sequence(conll, actions, encoder_features, transition_model,
        relation_model=None, direction_model=None, more_context=False, use_cuda=False):
  """Execute a given action sequence, also find best relations."""
  stack = data_utils.ParseForest([])
  buf = data_utils.ParseForest([conll[0]])
  buffer_index = 0
  sent_length = len(conll)

  labels = []
  transition_logits = []
  direction_logits = []
  relation_logits = []
  normalize = nn.Sigmoid()

  for action in actions:
    transition_logit = None
    relation_logit = None
    direction_logit = None

    label = -1
    s0 = stack.roots[-1].id if len(stack) > 0 else 0
    s1 = stack.roots[-2].id if len(stack) > 1 else 0
    s2 = stack.roots[-3].id if len(stack) > 2 else 0

    if len(stack) > 1: # allowed to ra or la
      position = nn_utils.extract_feature_positions(buffer_index, s0, s1, s2, more_context)
      features = nn_utils.select_features(encoder_features, position, use_cuda)
      
      if direction_model is not None: 
        transition_logit = transition_model(features)
        if buffer_index == sent_length:
          assert action == data_utils._RA
          sh_action = data_utils._SRE 
        else: 
          transition_sigmoid = normalize(transition_logit)
          transition_sigmoid_np = transition_sigmoid.type(torch.FloatTensor).data.numpy()
          if action == data_utils._SH:
            sh_action = data_utils._SSH
          else:
            sh_action = data_utils._SRE
        if sh_action == data_utils._SRE:
          direction_logit = direction_model(features)  
          direction_sigmoid = normalize(direction_logit)
          direction_sigmoid_np = direction_sigmoid.type(torch.FloatTensor).data.numpy()
          if buffer_index == sent_length:
            assert action == data_utils._RA
            direc = data_utils._DRA
          elif action == data_utils._LA:
            direc = data_utils._DLA  
          else:
            direc = data_utils._DRA
          direction_logits.append(direction_logit)
        else:
          assert action == data_utils._SH
          direction_logits.append(None)
      else:
        transition_logit = transition_model(features)
        transition_logit_np = transition_logit.type(torch.FloatTensor).data.numpy()
        if buffer_index == sent_length:
          assert action == data_utils._RA
      
      if relation_model is not None and action != data_utils._SH:
        relation_logit = relation_model(features)      
        relation_logit_np = relation_logit.type(torch.FloatTensor).data.numpy()
        label = int(relation_logit_np.argmax(axis=1)[0])
      
    transition_logits.append(transition_logit)
    if relation_model is not None:
      relation_logits.append(relation_logit)
      labels.append(label)

    # excecute action
    if action == data_utils._SH:
      stack.roots.append(buf.roots[0]) 
      buffer_index += 1
      if buffer_index == sent_length:
        buf = data_utils.ParseForest([])
      else:
        buf = data_utils.ParseForest([conll[buffer_index]])
    else:  
      assert len(stack) > 0
      child = stack.roots.pop()
      if action == data_utils._LA:
        buf.roots[0].children.append(child) 
        conll[child.id].pred_parent_id = buf.roots[0].id
      else:
        stack.roots[-1].children.append(child)
        conll[child.id].pred_parent_id = stack.roots[-1].id
      if relation_model is not None:
        conll[child.id].pred_relation_ind = label

  return conll, transition_logits, direction_logits, actions, relation_logits, labels


def viterbi_decode(conll, encoder_features, transition_model,
        word_model=None, direction_model=None, relation_model=None,
        use_cuda=False):
  sent_length = len(conll) # includes root, but not EOS
  log_normalize = nn.LogSoftmax()
  binary_normalize = nn.Sigmoid()

  # compute all sh/re and word probabilities
  seq_length = sent_length + 1
  reduce_probs = np.zeros([seq_length, seq_length])
  ra_probs = np.zeros([seq_length, seq_length])
  word_log_probs = np.empty([sent_length, sent_length])
  word_log_probs.fill(-np.inf)

  # batch feature computation
  features = nn_utils.batch_feature_selection(encoder_features, seq_length, use_cuda)
  re_probs_list = nn_utils.to_numpy(binary_normalize(transition_model(features)))
  ra_probs_list = nn_utils.to_numpy(binary_normalize(direction_model(features)))
  if word_model is not None:
    word_dist = log_normalize(word_model(features))

  counter = 0 
  for i in range(seq_length-1):
    for j in range(i+1, seq_length):
      reduce_probs[i, j] = re_probs_list[counter, 0]
      ra_probs[i, j] = ra_probs_list[counter, 0]
      if word_model is not None and j < sent_length:
        if j < sent_length - 1:
          word_id = conll[j+1].word_id 
        else:
          word_id = data_utils._EOS
        word_log_probs[i, j] = nn_utils.to_numpy(word_dist[counter, word_id])
      counter += 1

  table_size = sent_length + 1
  table = np.empty([table_size, table_size, table_size])
  table.fill(-np.inf) # log probabilities
  split_indexes = np.zeros((table_size, table_size, table_size), dtype=np.int)
  directions = np.zeros((table_size, table_size, table_size), dtype=np.int)
  directions.fill(data_utils._DRA) # default direction

  # first word prob 
  if word_model is not None:
    init_features = nn_utils.select_features(encoder_features, [0, 0], use_cuda)
    word_dist = log_normalize(word_model(init_features).view(-1))
    table[0, 0, 1] = nn_utils.to_numpy(word_dist[conll[1].word_id])
  else:
    table[0, 0, 1] = 0

  for j in range(2, sent_length+1):
    for i in range(j-1):
      score = np.log(1 - reduce_probs[i, j-1]) 
      if word_model is not None:
        score += word_log_probs[i, j-1]
      table[i, j-1, j] = score
    for i in range(j-2, -1, -1):
      for l in range(max(i, 1)): # l=0 if i=0
        block_scores = [] # adjust indexes after collecting scores
        block_directions = []
        for k in range(i+1, j):
          score = table[l, i, k] + table[i, k, j] + np.log(reduce_probs[k, j])
          if direction_model is not None:
            ra_prob = ra_probs[k, j]
            if ra_prob > 0.5 or j == sent_length:
              score += np.log(ra_prob)
              block_directions.append(data_utils._DRA)
            else:
              score += np.log(1-ra_prob)
              block_directions.append(data_utils._DLA)
          block_scores.append(score)
        ind = np.argmax(block_scores)
        k = ind + i + 1
        table[l, i, j] = block_scores[ind]
        split_indexes[l, i, j] = k
        if direction_model is not None:
          directions[l, i, j] = block_directions[ind]
  actions = backtrack_path(table, split_indexes, directions, 0, 0, sent_length)
  print(table[0, 0, sent_length]) # scores should match
  return decode_action_sequence(conll, actions, encoder_features,
          transition_model, relation_model, direction_model, use_cuda=use_cuda)

#TODO have versions with and without scoring
def greedy_decode(conll, encoder_features, transition_model,
        relation_model=None, direction_model=None, more_context=False,
        use_cuda=False):
  stack = data_utils.ParseForest([])
  buf = data_utils.ParseForest([conll[0]])
  buffer_index = 0
  sent_length = len(conll)

  actions = []
  labels = []
  transition_logits = []
  direction_logits = []
  relation_logits = []
  normalize = nn.Sigmoid()

  while buffer_index < sent_length or len(stack) > 1:
    transition_logit = None
    relation_logit = None
    direction_logit = None

    action = data_utils._SH
    label = -1
    s0 = stack.roots[-1].id if len(stack) > 0 else 0
    s1 = stack.roots[-2].id if len(stack) > 1 else 0
    s2 = stack.roots[-3].id if len(stack) > 2 else 0

    if len(stack) > 1: # allowed to ra or la
      position = nn_utils.extract_feature_positions(buffer_index, s0, s1, s2, more_context)
      features = nn_utils.select_features(encoder_features, position, use_cuda)
      
      #TODO rather score transition and relation jointly for greedy choice
      if direction_model is not None: # find action from decomposed 
        transition_logit = transition_model(features)
        if buffer_index == sent_length:
          sh_action = data_utils._SRE 
        else: #TODO check, but instead of sigmoid can just test > 0
          transition_sigmoid = normalize(transition_logit)
          transition_sigmoid_np = transition_sigmoid.type(torch.FloatTensor).data.numpy()
          sh_action = int(np.round(transition_sigmoid_np)[0])
        if sh_action == data_utils._SRE:
          direction_logit = direction_model(features)  
          direction_sigmoid = normalize(direction_logit)
          direction_sigmoid_np = direction_sigmoid.type(torch.FloatTensor).data.numpy()
          if buffer_index == sent_length:
            direc = data_utils._DRA
          else:
            direc = int(np.round(direction_sigmoid_np)[0])
          if direc == data_utils._DLA:
            action = data_utils._LA
          else:
            action = data_utils._RA
          direction_logits.append(direction_logit)
        else:
          action = data_utils._SH
          direction_logits.append(None)
      else:
        transition_logit = transition_model(features)
        transition_logit_np = transition_logit.type(torch.FloatTensor).data.numpy()
        if buffer_index == sent_length:
          action = data_utils._RA
        else:
          action = int(transition_logit_np.argmax(axis=1)[0])
      
      if relation_model is not None and action != data_utils._SH:
        relation_logit = relation_model(features)      
        relation_logit_np = relation_logit.type(torch.FloatTensor).data.numpy()
        label = int(relation_logit_np.argmax(axis=1)[0])
      
    actions.append(action)
    transition_logits.append(transition_logit)
    if relation_model is not None:
      relation_logits.append(relation_logit)
      labels.append(label)

    # excecute action
    if action == data_utils._SH:
      stack.roots.append(buf.roots[0]) 
      buffer_index += 1
      if buffer_index == sent_length:
        buf = data_utils.ParseForest([])
      else:
        buf = data_utils.ParseForest([conll[buffer_index]])
    else:  
      assert len(stack) > 0
      child = stack.roots.pop()
      if action == data_utils._LA:
        buf.roots[0].children.append(child) 
        conll[child.id].pred_parent_id = buf.roots[0].id
      else:
        stack.roots[-1].children.append(child)
        conll[child.id].pred_parent_id = stack.roots[-1].id
      if relation_model is not None:
        conll[child.id].pred_relation_ind = label

  return conll, transition_logits, direction_logits, actions, relation_logits, labels
 

def inside_score_decode(conll, encoder_features, transition_model, word_model,
    use_cuda=False):
  '''Compute inside score for testing, not training.'''
  assert word_model is not None
  sent_length = len(conll) # includes root, but not EOS
  log_normalize = nn.LogSoftmax()
  binary_normalize = nn.Sigmoid()

# compute all sh/re and word probabilities
  seq_length = sent_length + 1
  reduce_probs = np.zeros([seq_length, seq_length])
  ra_probs = np.zeros([seq_length, seq_length])
  word_log_probs = np.empty([sent_length, sent_length])
  word_log_probs.fill(-np.inf)

  # batch feature computation
  features = nn_utils.batch_feature_selection(encoder_features, seq_length, use_cuda)
  re_probs_list = nn_utils.to_numpy(binary_normalize(transition_model(features)))
  word_dist = log_normalize(word_model(features))

  counter = 0 
  for i in range(seq_length-1):
    for j in range(i+1, seq_length):
      reduce_probs[i, j] = re_probs_list[counter, 0]
      if word_model is not None and j < sent_length:
        if j < sent_length - 1:
          word_id = conll[j+1].word_id 
        else:
          word_id = data_utils._EOS
        word_log_probs[i, j] = nn_utils.to_numpy(word_dist[counter, word_id])
      counter += 1

  table_size = sent_length + 1
  table = np.empty([table_size, table_size, table_size])
  table.fill(-np.inf) # log probabilities

  # first word prob 
  if word_model is not None:
    init_features = nn_utils.select_features(encoder_features, [0, 0], use_cuda)
    word_dist = log_normalize(word_model(init_features).view(-1))
    table[0, 0, 1] = nn_utils.to_numpy(word_dist[conll[1].word_id])
  else:
    table[0, 0, 1] = 0

  for j in range(2, sent_length+1):
    for i in range(j-1):
      score = np.log(1 - reduce_probs[i, j-1]) 
      if word_model is not None:
        score += word_log_probs[i, j-1]
      table[i, j-1, j] = score
    for i in range(j-2, -1, -1):
      for l in range(max(i, 1)): # l=0 if i=0
        score = -np.inf
        for k in range(i+1, j):
          item_score = (table[l, i, k] + table[i, k, j] 
                        + np.log(reduce_probs[k, j]))
          score = np.logaddexp(score, item_score)
        table[l, i, j] = score
  return table[0, 0, sent_length]


def train_oracle(conll, encoder_features, more_context=False, use_cuda=False):
  # Training oracle for single parsed sentence
  stack = data_utils.ParseForest([])
  buf = data_utils.ParseForest([conll[0]])
  buffer_index = 0
  sent_length = len(conll)

  num_children = [0 for _ in conll]
  for token in conll:
    num_children[token.parent_id] += 1

  actions = []
  labels = []
  words = []
  features = []

  while buffer_index < sent_length or len(stack) > 1:
    feature = None
    action = data_utils._SH
    s0 = stack.roots[-1].id if len(stack) > 0 else 0
    s1 = stack.roots[-2].id if len(stack) > 1 else 0
    s2 = stack.roots[-3].id if len(stack) > 2 else 0

    if len(stack) > 1: # allowed to ra or la
      if buffer_index == sent_length:
        action = data_utils._RA
      else:
        if stack.roots[-1].parent_id == buffer_index:
          if len(stack.roots[-1].children) == num_children[s0]:
            action = data_utils._LA 
        elif stack.roots[-1].parent_id == s1:
          if len(stack.roots[-1].children) == num_children[s0]:
            action = data_utils._RA 
 
    position = nn_utils.extract_feature_positions(buffer_index, s0, s1, s2, more_context) 
    feature = nn_utils.select_features(encoder_features, position, use_cuda)
    features.append(feature)
     
    label = -1 if action == data_utils._SH else stack.roots[-1].relation_id
    if action == data_utils._SH:
      if buffer_index+1 < sent_length:
        word = conll[buffer_index+1].word_id
      else:
        word = data_utils._EOS
    else:
      word = -1
        
    actions.append(action)
    labels.append(label)
    words.append(word)

    # excecute action
    if action == data_utils._SH:
      stack.roots.append(buf.roots[0]) 
      buffer_index += 1
      if buffer_index == sent_length:
        buf = data_utils.ParseForest([])
      else:
        buf = data_utils.ParseForest([conll[buffer_index]])
    else:  
      assert len(stack) > 0
      child = stack.roots.pop()
      if action == data_utils._LA:
        buf.roots[0].children.append(child) 
      else:
        stack.roots[-1].children.append(child)
  return actions, words, labels, features


## originally inside main

#TODO need something like this for other parsers too
def score(args, dev_sentences, test_sentences, word_vocab, pos_vocab, 
          rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  num_transitions = 3
  num_features = 4 if args.use_more_features else 2
  batch_size = args.batch_size

  working_path = args.working_dir + '/'
  print('Loading models')
  # Load models. TODO only load parameters
  model_fn = working_path + args.save_model + '_encoder.pt'
  with open(model_fn, 'rb') as f:
    encoder_model = torch.load(f)
  model_fn = working_path + args.save_model + '_transition.pt'
  with open(model_fn, 'rb') as f:
    transition_model = torch.load(f)
  if args.predict_relations:
    model_fn = working_path + args.save_model + '_relation.pt'
    with open(model_fn, 'rb') as f:
      relation_model = torch.load(f)
  if args.generative:
    model_fn = working_path + args.save_model + '_word.pt'
    with open(model_fn, 'rb') as f:
      word_model = torch.load(f)
  if args.decompose_actions:
    model_fn = working_path + args.save_model + '_direction.pt'
    with open(model_fn, 'rb') as f:
      direction_model = torch.load(f)

  #TODO not sure if this is neccessary
  if args.cuda:
    encoder_model.cuda()
    transition_model.cuda()
    if args.predict_relations:
      relation_model.cuda()
    if args.generative:
      word_model.cuda()
    if args.decompose_actions:
      direction_model.cuda()

  criterion = nn.CrossEntropyLoss(size_average=args.criterion_size_average)
  binary_criterion = nn.BCELoss(size_average=args.criterion_size_average)

  print('Done loading models')

  val_batch_size = 1
  total_loss = 0
  total_length = 0
  conll_predicted = []
  dev_losses = []
  decode_start_time = time.time()

  encoder_model.eval()
  normalize = nn.Sigmoid() 
  if not args.generative:
    word_model = None

  for val_sent in dev_sentences:
    sentence_loss = 0
    sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda, evaluation=True)
    encoder_state = encoder_model.init_hidden(val_batch_size)
    encoder_output = encoder_model(sentence_data, encoder_state)
    total_length += len(val_sent) - 1 

    if args.decompose_actions and args.inside_decode: 
      inside_score = inside_score_decode(val_sent.conll, encoder_output,
            transition_model, word_model, use_cuda=args.cuda)
      #inside_score.backward()
      #score = nn_utils.to_numpy(inside_score)[0]  
      score = inside_score
      print(score)
      dev_losses.append(score)
      total_loss += score
    #else:
      #TODO greedy score

  val_loss = - total_loss / total_length
  print('-' * 89)
  print('| decoding time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
  print('-' * 89)


def decode(args, dev_sentences, test_sentences, word_vocab, pos_vocab, 
           rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  num_transitions = 3
  num_features = 4 if args.use_more_features else 2
  batch_size = args.batch_size

  print('Loading models')
  # Load models. TODO only load parameters
  model_fn = working_path + args.save_model + '_encoder.pt'
  with open(model_fn, 'rb') as f:
    encoder_model = torch.load(f)
  model_fn = working_path + args.save_model + '_transition.pt'
  with open(model_fn, 'rb') as f:
    transition_model = torch.load(f)
  if args.predict_relations:
    model_fn = working_path + args.save_model + '_relation.pt'
    with open(model_fn, 'rb') as f:
      relation_model = torch.load(f)
  if args.generative:
    model_fn = working_path + args.save_model + '_word.pt'
    with open(model_fn, 'rb') as f:
      word_model = torch.load(f)
  if args.decompose_actions:
    model_fn = working_path + args.save_model + '_direction.pt'
    with open(model_fn, 'rb') as f:
      direction_model = torch.load(f)

  #TODO not sure if this is neccessary
  if args.cuda:
    encoder_model.cuda()
    transition_model.cuda()
    if args.predict_relations:
      relation_model.cuda()
    if args.generative:
      word_model.cuda()
    if args.decompose_actions:
      direction_model.cuda()

  criterion = nn.CrossEntropyLoss(size_average=args.criterion_size_average)
  binary_criterion = nn.BCELoss(size_average=args.criterion_size_average)

  print('Done loading models')

  val_batch_size = 1
  total_loss = 0
  total_length = 0
  conll_predicted = []
  dev_losses = []
  decode_start_time = time.time()

  encoder_model.eval()
  normalize = nn.Sigmoid() 
  if not args.generative:
    word_model = None

  for val_sent in dev_sentences:
    sentence_loss = 0
    sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda, evaluation=True)
    encoder_state = encoder_model.init_hidden(val_batch_size)
    encoder_output = encoder_model(sentence_data, encoder_state)
    total_length += len(val_sent) - 1 

    #TODO evaluate word prediction for generative model
    if args.decompose_actions:
      if args.viterbi_decode:
        predict, transition_logits, direction_logits, actions, relation_logits, labels = viterbi_decode(
            val_sent.conll, encoder_output, transition_model, word_model,
            direction_model, relation_model, use_cuda=args.cuda)
        print(actions)
      else:
        predict, transition_logits, direction_logits, actions, relation_logits, labels = greedy_decode(
          val_sent.conll, encoder_output, transition_model, relation_model,
          direction_model, use_cuda=args.cuda)
    else:
      predict, transition_logits, _, actions, relation_logits, labels = greedy_decode(
        val_sent.conll, encoder_output, transition_model, relation_model,
        more_context=args.use_more_features, use_cuda=args.cuda)

    for j, token in enumerate(predict):
      # Convert labels to str
      if j > 0 and relation_model is not None and token.pred_relation_ind >= 0:
        predict[j].pred_relation = rel_vocab.get_word(token.pred_relation_ind)
    conll_predicted.append(predict) 

    if args.decompose_actions:
      actions, directions = decompose_transitions(actions)

    # Filter out Nones to get examples for loss, then concatenate
    if args.decompose_actions:
      transition_output, action_var = nn_utils.filter_logits(transition_logits,
          actions, float_var=True, use_cuda=args.cuda)
      direction_output, dir_var = nn_utils.filter_logits(direction_logits,
        directions, float_var=True, use_cuda=args.cuda)
    else:
      transition_output, action_var = nn_utils.filter_logits(transition_logits,
         actions, use_cuda=args.cuda)

    if transition_output is not None:
      if args.decompose_actions:
        tr_loss = binary_criterion(normalize(transition_output.view(-1)),
                                action_var).data
        #if math.isnan(tr_loss[0]):
        #  print('Transition loss')
        #  print(transition_output)
        #  print(action_var)
        sentence_loss += tr_loss
        if direction_output is not None:
          dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                                  dir_var).data
          #if math.isnan(dir_loss[0]):
          #  print('Direction loss')
          #  print(direction_output)
          #  print(dir_var)
          sentence_loss += dir_loss
      else:
        sentence_loss += criterion(transition_output.view(-1, num_transitions),
                                action_var).data

      if args.predict_relations:
        relation_output, label_var = nn_utils.filter_logits(relation_logits,
            labels, use_cuda=args.cuda)
        if relation_output is not None:
          sentence_loss += criterion(relation_output.view(-1, num_relations),
                                  label_var).data
    total_loss += sentence_loss
    dev_losses.append(sentence_loss[0])

  with open(working_path + args.dev_name + '.score', 'w') as fh:
    for loss in dev_losses:
      fh.write(str(loss) + '\n')
  data_utils.write_conll(working_path + args.dev_name + '.output.conll', conll_predicted)
  val_loss = total_loss[0] / total_length
  print('-' * 89)
  print('| decoding time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
  print('-' * 89)


def train(args, sentences, dev_sentences, test_sentences, word_vocab, 
          pos_vocab, rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  num_transitions = 3
  num_features = 2 if args.decompose_actions else 4

  batch_size = args.batch_size

  # Build the model
  encoder_model = rnn_encoder.RNNEncoder(vocab_size, 
      args.embedding_size, args.hidden_size, args.num_layers, args.dropout,
      args.bidirectional, args.cuda)

  #TODO factorize features to work with dynamic program
  feature_size = (args.hidden_size*2 if args.bidirectional 
                  else args.hidden_size)
  #TODO add option for decomposition depending on how it goes
  if args.decompose_actions:
    transition_model = binary_classifier.BinaryClassifier(num_features, 
      feature_size, args.hidden_size, args.cuda) 
    direction_model = binary_classifier.BinaryClassifier(num_features, 
      feature_size, args.hidden_size, args.cuda) 
  else:
    transition_model = classifier.Classifier(num_features, feature_size, 
      args.hidden_size, num_transitions, args.cuda) 
  if args.predict_relations:
    relation_model = classifier.Classifier(num_features, feature_size, 
        args.hidden_size, num_relations, args.cuda)
  else:
    relation_model = None
  if args.generative:
    word_model = classifier.Classifier(num_features, feature_size,
            args.hidden_size, vocab_size, args.cuda)

  if args.cuda:
    encoder_model.cuda()
    transition_model.cuda()
    if args.predict_relations:
      relation_model.cuda()
    if args.generative:
      word_model.cuda()
    if args.decompose_actions:
      direction_model.cuda()
  criterion = nn.CrossEntropyLoss(size_average=args.criterion_size_average)
  binary_criterion = nn.BCELoss(size_average=args.criterion_size_average)

  params = (list(encoder_model.parameters()) 
            + list(transition_model.parameters()))
  if args.predict_relations:
    params += list(relation_model.parameters())
  if args.generative:
    params += list(word_model.parameters())
  if args.decompose_actions:
    params += list(direction_model.parameters())

  optimizer = optim.Adam(params, lr=args.lr)
 
  prev_val_loss = None
  for epoch in range(1, args.epochs+1):
    print('Start training epoch %d' % epoch)
    epoch_start_time = time.time()
    
    random.shuffle(sentences)
    #sentences.sort(key=len) 
    #TODO batch for RNN encoder

    total_loss = 0 
    global_loss = 0 
    total_num_tokens = 0 
    global_num_tokens = 0 
    encoder_model.train()

    start_time = time.time()
    for i, train_sent in enumerate(sentences):
      # Training loop

      # sentence encoder
      sentence_data = nn_utils.get_sentence_batch(train_sent, args.cuda)
      encoder_model.zero_grad()
      transition_model.zero_grad()
      if args.predict_relations:
        relation_model.zero_grad()
      if args.generative:
        word_model.zero_grad()
      normalize = nn.Sigmoid()

      encoder_state = encoder_model.init_hidden(batch_size)
      encoder_output = encoder_model(sentence_data, encoder_state)

      actions, words, labels, features = train_oracle(train_sent.conll, 
              encoder_output, args.use_more_features, use_cuda=args.cuda)
      
      if args.decompose_actions:
        actions, directions = decompose_transitions(actions)

      # when will the direction logits be none? -> from features
      # but 2-features will be used for both sh and re

      # Filter out Nones and concatenate to get training examples
      if args.decompose_actions:
        transition_logits = [transition_model(feat) if feat is not None
                             else None for feat in features] 
        direction_logits = [direction_model(feat) 
                             if feat is not None and direct is not None
                           else None for feat, direct in zip(features,
                               directions)] 
        direction_output, dir_var = nn_utils.filter_logits(direction_logits,
            directions, float_var=True, use_cuda=args.cuda)
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
            actions, float_var=True, use_cuda=args.cuda)
      else:
        transition_logits = [transition_model(feat) if feat is not None
                             else None for feat in features] 
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
            actions, use_cuda=args.cuda)
      
      if args.predict_relations:
        relation_logits = []
        for feat, action in zip(features, actions):
          if action != data_utils._SH and feat is not None:
            relation_logits.append(relation_model(feat))
          else:
            relation_logits.append(None)

        relation_output, label_var = nn_utils.filter_logits(relation_logits, labels, use_cuda=args.cuda)

      if args.generative:
        gen_word_logits = []
        for feat, action, word in zip(features, actions, words):
          if action == data_utils._SH:
            assert word >= 0
            gen_word_logits.append(word_model(feat))
          else:
            gen_word_logits.append(None)
        gen_word_output, word_var = nn_utils.filter_logits(gen_word_logits, words, use_cuda=args.cuda)

      loss = None
      if args.decompose_actions:
        if transition_output is not None:
          loss = binary_criterion(normalize(transition_output.view(-1)),
                         action_var)
          if direction_logits is not None:
            dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                         dir_var)
            loss = loss + dir_loss if loss is not None else dir_loss

      else:
        if transition_output is not None:
          loss = criterion(transition_output.view(-1, num_transitions),
                         action_var)
       
      if args.predict_relations and relation_output is not None:
          rel_loss = criterion(relation_output.view(-1, num_relations), 
                               label_var)
          loss = loss + rel_loss if loss is not None else rel_loss

      if args.generative and gen_word_output is not None:
          word_loss = criterion(gen_word_output.view(-1, vocab_size),
                                word_var)
          loss = loss + word_loss if loss is not None else word_loss

      total_num_tokens += len(train_sent) - 1 
      global_num_tokens += len(train_sent) - 1 
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
    print('| end of epoch {:3d} | {:5d} batches | tokens {:5d} | loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, len(sentences), global_num_tokens,
        avg_global_loss, math.exp(avg_global_loss)))

    val_batch_size = 1
    total_loss = 0
    total_length = 0
    conll_predicted = []
    decode_start_time = time.time()

    encoder_model.eval()
    normalize = nn.Sigmoid() 
    if not args.generative:
      word_model = None
  
    print('Decoding dev sentences')
    for val_sent in dev_sentences:
      sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda, evaluation=True)
      encoder_state = encoder_model.init_hidden(val_batch_size)
      encoder_output = encoder_model(sentence_data, encoder_state)
  
      if args.decompose_actions:
        if args.viterbi_decode:
          predict, transition_logits, direction_logits, actions, relation_logits, labels = viterbi_decode(
              val_sent.conll, encoder_output, transition_model, word_model,
              direction_model, relation_model)
          print(actions)
        else:
          predict, transition_logits, direction_logits, actions, relation_logits, labels = greedy_decode(
            val_sent.conll, encoder_output, transition_model, relation_model,
            direction_model, use_cuda=args.cuda)
      else:
        predict, transition_logits, _, actions, relation_logits, labels = greedy_decode(
          val_sent.conll, encoder_output, transition_model, relation_model,
          more_context=args.use_more_features, use_cuda=args.cuda)

      #TODO need to compute word probabilities here
      for j, token in enumerate(predict):
        # Convert labels to str
        if j > 0 and relation_model is not None and token.pred_relation_ind >= 0:
          predict[j].pred_relation = rel_vocab.get_word(token.pred_relation_ind)
      conll_predicted.append(predict) 

      if args.decompose_actions:
        actions, directions = decompose_transitions(actions)

      # Filter out Nones to get examples for loss, then concatenate
      if args.decompose_actions:
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
            actions, float_var=True, use_cuda=args.cuda)
        direction_output, dir_var = nn_utils.filter_logits(direction_logits,
          directions, float_var=True, use_cuda=args.cuda)
      else:
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
           actions, use_cuda=args.cuda)

      total_length += len(val_sent) - 1
      if transition_output is not None:
        if args.decompose_actions:
          tr_loss = binary_criterion(normalize(transition_output.view(-1)),
                                  action_var).data
          total_loss += tr_loss
          if direction_output is not None:
            dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                                    dir_var).data
            total_loss += dir_loss
        else:
          total_loss += criterion(transition_output.view(-1, num_transitions),
                                  action_var).data

        if args.predict_relations:
          relation_output, label_var = nn_utils.filter_logits(relation_logits, labels, use_cuda=args.cuda)
          if relation_output is not None:
            total_loss += criterion(relation_output.view(-1, num_relations),
                                    label_var).data

    working_path = args.working_dir + '/'
    data_utils.write_conll(working_path + args.dev_name + '.' + str(epoch) + '.output.conll', conll_predicted)
    val_loss = total_loss[0] / total_length
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | {:5d} tokens | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
        epoch, (time.time() - epoch_start_time), total_length, val_loss,
        math.exp(val_loss)))
    print('decoding time: {:5.2f}s'.format(time.time() - decode_start_time))
    print('-' * 89)

    # save the model #TODO save only parameters
    if args.save_model != '':
      model_fn = working_path + args.save_model + '_encoder.pt'
      with open(model_fn, 'wb') as f:
        torch.save(encoder_model, f)
      model_fn = working_path + args.save_model + '_transition.pt'
      with open(model_fn, 'wb') as f:
        torch.save(transition_model, f)
      if relation_model is not None:
        model_fn = working_path + args.save_model + '_relation.pt'
        with open(model_fn, 'wb') as f:
          torch.save(relation_model, f)
      if args.generative:
        model_fn = working_path + args.save_model + '_word.pt'
        with open(model_fn, 'wb') as f:
          torch.save(word_model, f)
      if args.decompose_actions:
        model_fn = working_path + args.save_model + '_direction.pt'
        with open(model_fn, 'wb') as f:
          torch.save(direction_model, f)


