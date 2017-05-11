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
      embedding_size, hidden_size, num_layers, dropout, bidirectional, 
      more_context, predict_relations, generative, decompose_actions,
      batch_size, use_cuda):
    assert not decompose_actions and not more_context
    num_transitions = 3
    num_features = 2
    super(ArcEagerTransitionSystem, self).__init__(vocab_size, num_relations,
        num_features, num_transitions,
        embedding_size, hidden_size, num_layers, dropout, bidirectional,
        predict_relations, generative, decompose_actions,
        batch_size, use_cuda)
    self.more_context = False
    self.generate_actions = [data_utils._SH, data_utils._RA]

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


  def greedy_decode(self, conll, encoder_features):
    stack = data_utils.ParseForest([])
    stack_has_parent = []
    stack_parent_relation = []
    buf = data_utils.ParseForest([conll[0]])
    buffer_index = 0
    sent_length = len(conll)
    #print(sent_length)

    actions = []
    labels = []
    transition_logits = []
    direction_logits = []
    relation_logits = []
    normalize = nn.Sigmoid()

    while buffer_index < sent_length or len(stack) > 1:
      transition_logit = None
      relation_logit = None

      action = data_utils._SH
      pred_action = data_utils._ESH
      label = -1
      s0 = stack.roots[-1].id if len(stack) > 0 else 0

      if len(stack) > 0: # allowed to ra or la
        position = nn_utils.extract_feature_positions(buffer_index, s0)
        features = nn_utils.select_features(encoder_features, position, self.use_cuda)
        transition_logit = self.transition_model(features)
        transition_logit_np = transition_logit.type(torch.FloatTensor).data.numpy()
        if buffer_index == sent_length:
          pred_action = data_utils._ERE
        else:
          #if stack_has_parent[-1]:
          #  transition_logit_np[0, data_utils._LA] = -np.inf
          #else:
          #  transition_logit_np[0, data_utils._RE] = -np.inf
          pred_action = int(transition_logit_np.argmax(axis=1)[0])
          transition_logits.append(transition_logit)

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
        
      transition_logits.append(transition_logit)
      actions.append(pred_action) 
      if self.relation_model is not None:
        relation_logits.append(relation_logit)
        labels.append(label)

      # excecute action
      if action == data_utils._SH or action == data_utils._RA:
        child = buf.roots[0]
        if action == data_utils._RA:
          # this duplication might cause a problem
          #stack.roots[-1].children.append(child) 
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
    #print(actions)
    return conll, transition_logits, None, actions, relation_logits, labels
   

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
      feature = nn_utils.select_features(encoder_features, position, self.use_cuda)
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

