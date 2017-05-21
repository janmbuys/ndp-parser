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

class ArcEagerTransitionSystem(tr.TransitionSystem):
  def __init__(self, vocab_size, num_relations,
      embedding_size, hidden_size, num_layers, dropout, init_weight_range, 
      bidirectional, more_context, predict_relations, generative, 
      decompose_actions, batch_size, use_cuda, model_path='', load_model=False):
    assert not decompose_actions and not more_context
    num_transitions = 3
    num_features = 3 # now including headed feature
    super(ArcEagerTransitionSystem, self).__init__(vocab_size, num_relations,
        num_features, num_transitions, embedding_size, hidden_size, 
        num_layers, dropout, init_weight_range, bidirectional,
        predict_relations, generative, decompose_actions,
        batch_size, use_cuda, model_path, load_model)
    self.more_context = False
    self.generate_actions = [data_utils._SH, data_utils._RA]

    # TODO store and load
    # embed binary features
    self.embed_headed = nn.Embedding(2, feature_size)
    self.embed_shift = nn.Embedding(2, feature_size)
    self.init_weights(init_weight_range)
     
    if use_cuda:
      self.embed_headed.cuda()
      self.embed_shift.cuda()

  def init_weights(self, initrange=0.1):
    self.embed_headed.weight.data.uniform_(-initrange, initrange)
    self.embed_shift.weight.data.uniform_(-initrange, initrange)

  def map_transitions(self, actions):
    nactions = []
    for action in actions:
      if action == data_utils._SH:
        nactions.append(data_utils._ESH)
      elif action == data_utils._RA:
        nactions.append(data_utils._ERA)
      else:
        nactions.append(data_utils._ERE)
    return nactions 

  def decompose_transitions(self, actions):
    # Note: Don't actually do this now.
    stackops = []
    arcops = []
    
    for action in actions:
      if action == data_utils._SH:
        stackops.append(data_utils._SSH)
        arcops.append(data_utils._LSH)
      elif action == data_utils._RA:
        stackops.append(data_utils._SSH)
        arcops.append(data_utils._LRA)
      elif action == data_utils._LA:
        stackops.append(data_utils._SRE)
        arcops.append(data_utils._ULA)
      else:
        stackops.append(data_utils._SRE)
        arcops.append(data_utils._URE)
   
    return stackops, arcops

  def greedy_decode(self, conll, encoder_features, given_actions=None):
    stack = data_utils.ParseForest([])
    stack_has_parent = []
    stack_parent_relation = []
    buf = data_utils.ParseForest([conll[0]])
    buffer_index = 0
    sent_length = len(conll)

    num_actions = 0
    predicted_actions = []
    labels = []
    words = []
    transition_logits = []
    direction_logits = []
    gen_word_logits = []
    relation_logits = []
    normalize = nn.Sigmoid()

    while buffer_index < sent_length or len(stack) > 1:
      transition_logit = None
      relation_logit = None

      if given_actions is not None:
        assert len(given_actions) > num_actions
        action = given_actions[num_actions]
      else:
        action = data_utils._SH
      pred_action = data_utils._ESH

      label = -1
      s0 = stack.roots[-1].id if len(stack) > 0 else 0

      position = nn_utils.extract_feature_positions(buffer_index, s0)
      encoded_features = nn_utils.select_features(encoder_features, position, self.use_cuda)
      head_ind = torch.LongTensor([1 if stack_has_parent and stack_has_parent[-1] else 0]).view(1, 1)
      head_feat = self.embed_headed(nn_utils.to_var(head_ind, self.use_cuda)) 
      features = torch.cat((encoded_features, head_feat), 0)

      if len(stack) > 0: # allowed to ra or la
        transition_logit = self.transition_model(features)
        transition_logit_np = transition_logit.type(torch.FloatTensor).data.numpy()

        if buffer_index == sent_length:
          if given_actions is not None:
            assert action == data_utils._RE or action == data_utils._LA
          pred_action = data_utils._ERE
        else:
          if given_actions is not None:
            if action == data_utils._SH:
              pred_action = data_utils._ESH
            elif action == data_utils._RA:
              pred_action = data_utils._ERA
            else:
              pred_action = data_utils._ERE
          else:  
            pred_action = int(transition_logit_np.argmax(axis=1)[0])
          transition_logits.append(transition_logit)

        if given_actions is not None:
          if pred_action == data_utils._ERE:
            if stack_has_parent[-1] or buffer_index == sent_length:
              # TODO check if this will hold in practice
              assert action == data_utils._RE
            else:
              assert action == data_utils._LA
        else:      
          if pred_action == data_utils._ESH:
            action = data_utils._SH
          elif pred_action == data_utils._ERA:
            action = data_utils._RA
          elif stack_has_parent[-1] or buffer_index == sent_length:
            action = data_utils._RE
          else:
            action = data_utils._LA

        if (self.relation_model is not None
            and (action == data_utils._LA or action == data_utils._RA
                 or buffer_index == sent_length)):
          relation_logit = self.relation_model(features)      
          relation_logit_np = relation_logit.type(torch.FloatTensor).data.numpy()
          label = int(relation_logit_np.argmax(axis=1)[0])
        
      if self.word_model is not None and action == data_utils._SH:
        word_logit = self.word_model(features)
        gen_word_logits.append(word_logit)
        if buffer_index+1 < sent_length:
          word = conll[buffer_index+1].word_id
        else:
          word = data_utils._EOS
      else:
        word = -1
        if self.word_model is not None:
          gen_word_logits.append(None)

      if self.word_model is not None:
        words.append(word)

      transition_logits.append(transition_logit)
      num_actions += 1
      predicted_actions.append(pred_action) 
      if self.relation_model is not None:
        relation_logits.append(relation_logit)
        labels.append(label)

      # excecute action
      if action == data_utils._SH or action == data_utils._RA:
        child = buf.roots[0]
        if action == data_utils._RA:
          stack_has_parent.append(True)
        else:
          stack_has_parent.append(False)
        stack_parent_relation.append(label)

        stack.roots.append(child)
        buffer_index += 1
        if buffer_index == sent_length:
          buf = data_utils.ParseForest([])
        else:
          buf = data_utils.ParseForest([conll[buffer_index]])
      else:  
        assert len(stack) > 0
        child = stack.roots.pop()
        has_right_arc = stack_has_parent.pop()
        stack_label = stack_parent_relation.pop()
        if has_right_arc or buffer_index == sent_length: # reduce
          stack.roots[-1].children.append(child) 
          conll[child.id].pred_parent_id = stack.roots[-1].id
          if self.relation_model is not None:
            conll[child.id].pred_relation_ind = stack_label
        else: # left-arc
          buf.roots[0].children.append(child) 
          conll[child.id].pred_parent_id = buf.roots[0].id
          if self.relation_model is not None:
            conll[child.id].pred_relation_ind = label

    return conll, transition_logits, None, predicted_actions, relation_logits, labels, gen_word_logits, words
 
  def inside_score(self, conll, encoder_features):
    assert self.word_model is not None
    sent_length = len(conll) # includes root, but not eos
    eps = np.exp(-10) # used to avoid division by 0

    # compute all sh/re and word probabilities
    seq_length = sent_length + 1
    shift_log_probs = np.zeros([seq_length, seq_length, 2])
    ra_log_probs = np.zeros([seq_length, seq_length, 2])
    re_log_probs = np.zeros([seq_length, seq_length, 2])

    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    # batch feature computation
    enc_features = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda)
    num_items = enc_features.size()[0]

    # dim [num_items, 2 (headedness), batch_size, num_features, feature_size]
    enc_features.view(num_items, 1, 2, 1, self.feature_size).expand(
        num_items, 2, 2, 1, self.feature_size)

    # expand to add headedness feature
    head_ind0 = torch.LongTensor(num_items, 1, 1, 1).fill_(0)
    head_ind1 = torch.LongTensor(num_items, 1, 1, 1).fill_(1)
            
    head_feat0 = self.embed_headed(nn_utils.to_var(head_ind0, self.use_cuda)) 
    head_feat1 = self.embed_headed(nn_utils.to_var(head_ind1, self.use_cuda)) 
    head_features = torch.cat((head_feat0, head_feat1), 1)
    features = torch.cat((encoded_features, head_features), 3)

    tr_log_probs_list = nn_utils.to_numpy(self.log_normalize(
        self.transition_model(features))).view(-1, 2, self.num_transitions)
    word_dist = self.log_normalize(self.word_model(features))

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        for c in range(2):
          shift_log_probs[i, j, c] = tr_log_probs_list[counter, c, data_utils._ESH]
          re_log_probs[i, j, c] = np.logaddexp(
              tr_log_probs_list[counter, c, data_utils._LA], 
              tr_log_probs_list[counter, c, data_utils._RA])
          if j < sent_length:
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
    init_features = nn_utils.select_features(encoder_features, [0, 0, 0], self.use_cuda)
    word_dist = self.log_normalize(self.word_model(init_features).view(-1))
    table[0, 0, 1] = nn_utils.to_numpy(word_dist[conll[1].word_id])

    for j in range(2, sent_length+1):
      for i in range(j-1):
        score = shift_log_probs[i, j-1]
        score += word_log_probs[i, j-1]
        table[i, j-1, j] = score
      for i in range(j-2, -1, -1):
        for l in range(max(i, 1)): # l=0 if i=0
          block_score = -np.inf
          for k in range(i+1, j):
            item_score = table[l, i, k] + table[i, k, j] 
            if self.decompose_actions:
              item_score += np.log(reduce_probs[k, j] + eps)
            else:
              item_score += re_log_probs[k, j]
            block_score = np.logaddexp(block_score, item_score)
          table[l, i, j] = block_score
    return table[0, 0, sent_length]



  def oracle(self, conll, encoder_features):
    # Training oracle for single parsed sentence.
    stack = data_utils.ParseForest([])
    stack_has_parent = []
    buf = data_utils.ParseForest([conll[0]])
    buffer_index = 0
    sent_length = len(conll)

    late_reduce = True

    count_left_parent = [False for _ in conll]
    count_left_children = [0 for _ in conll]

    has_left_parent = [False for _ in conll]
    num_left_children = [0 for _ in conll]
    num_children = [0 for _ in conll]

    for token in conll:
      num_children[token.parent_id] += 1
      if token.id < token.parent_id:
        num_left_children[token.parent_id] += 1
      else:
        has_left_parent[token.id] = True

    actions = []
    labels = []
    words = []
    features = []

    while buffer_index < sent_length or len(stack) > 1:
      feature = None
      action = data_utils._SH
      s0 = stack.roots[-1].id if len(stack) > 0 else 0
      
      if len(stack) > 0: # allowed to ra or la
        if buffer_index == sent_length:
          action = data_utils._RE
        elif stack.roots[-1].parent_id == buffer_index:
          action = data_utils._LA 
        elif buf.roots[0].parent_id == stack.roots[-1].id:
          action = data_utils._RA 
        elif stack_has_parent[-1]:
          if (late_reduce and 
              ((has_left_parent[buffer_index] and not
                count_left_parent[buffer_index]) 
               or (count_left_children[buffer_index] <
                    num_left_children[buffer_index]))):
            action = data_utils._RE             
            assert len(stack.roots[-1].children) == num_children[s0]
          elif len(stack.roots[-1].children) == num_children[s0]:
            action = data_utils._RE 

      position = nn_utils.extract_feature_positions(buffer_index, s0) 
      encoded_feature = nn_utils.select_features(encoder_features, position, self.use_cuda)
     
      head_ind = torch.LongTensor([1 if stack_has_parent and stack_has_parent[-1] else 0]).view(1, 1)
      head_feat = self.embed_headed(nn_utils.to_var(head_ind, self.use_cuda)) 
      feature = torch.cat((encoded_feature, head_feat), 0)
      #TODO also add shift binary feature

      features.append(feature)
      
      if action == data_utils._LA:
        label = stack.roots[-1].relation_id
      elif action == data_utils._RA:
        label = buf.roots[0].relation_id
      else:
        label = -1 
      
      if action == data_utils._SH or action == data_utils._RA:
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
      if action == data_utils._SH or action == data_utils._RA:
        child = buf.roots[0]
        if action == data_utils._RA:
          # this duplication might cause a problem
          #stack.roots[-1].children.append(child) 
          count_left_parent[buffer_index] = True
          stack_has_parent.append(True)
        else:
          stack_has_parent.append(False)
        stack.roots.append(child)
        buffer_index += 1
        if buffer_index == sent_length:
          buf = data_utils.ParseForest([])
        else:
          buf = data_utils.ParseForest([conll[buffer_index]])
      else:  
        assert len(stack) > 0
        child = stack.roots.pop()
        has_right_arc = stack_has_parent.pop()
        if action == data_utils._LA:
          buf.roots[0].children.append(child) 
          count_left_children[buffer_index] += 1
        else:
          assert has_right_arc
          stack.roots[-1].children.append(child) 
    return actions, words, labels, features

