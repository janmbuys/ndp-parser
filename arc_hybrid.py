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
import transition_system as tr

class ArcHybridTransitionSystem(tr.TransitionSystem):
  def __init__(self, vocab_size, num_relations, num_features, num_transitions,
      embedding_size, hidden_size, num_layers, dropout, bidirectional, 
      more_context, predict_relations, generative, decompose_actions,
      batch_size, use_cuda):
    super(ArcHybridTransitionSystem, self).__init__(vocab_size, num_relations,
        num_features, num_transitions,
        embedding_size, hidden_size, num_layers, dropout, bidirectional,
        more_context, predict_relations, generative, decompose_actions,
        batch_size, use_cuda)
    self.generate_actions = [data_utils._SH]

  def decompose_transitions(self, actions):
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

  def _decode_action_sequence(self, conll, actions, encoder_features, 
      more_context=False, use_cuda=False):
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


  def viterbi_decode(self, conll, encoder_features, use_cuda=False):
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

    def backtrack_path(l, i, j):
      """ Find action sequence for best path. """
      #TODO add label prediction
      if i == j - 1:
        return [data_utils._SH]
      else:
        k = split_indexes[l, i, j]
        direct = directions[l, i, j]
        act = data_utils._LA if direct == data_utils._DLA else data_utils._RA
        return (backtrack_path(l, i, k) + backtrack_path(i, k, j) + [act])

    actions = backtrack_path(0, 0, sent_length)
    print(table[0, 0, sent_length]) # scores should match
    return self._decode_action_sequence(conll, actions, encoder_features,
            transition_model, relation_model, direction_model, use_cuda=use_cuda)

#TODO have versions with and without scoring
  def greedy_decode(self, conll, encoder_features, more_context=False,
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
   

  def inside_score_decode(self, conll, encoder_features, transition_model, word_model,
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


  def oracle(self, conll, encoder_features, more_context=False, use_cuda=False):
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

