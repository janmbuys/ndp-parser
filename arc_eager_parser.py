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

def train_oracle_ae(conll, encoder_features, more_context=False):
  # Arc eager training oracle for single parsed sentence
  stack = data_utils.ParseForest([])
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
      elif late_reduce:
        if not (has_left_parent[buffer_index] and not
                count_left_parent[buffer_index] or 
                (count_left_children[buffer_index] <
                    num_left_children[buffer_index])):
          action = data_utils._RE             
          assert len(stack.roots[-1].children) == num_children[s0]
      elif len(stack.roots[-1].children) == num_children[s0]:
        action = data_utils._RE 

    position = nn_utils.extract_feature_positions(buffer_index, s0) 
    feature = nn_utils.select_features(encoder_features, position, args.cuda)
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
        stack.roots[-1].children.append(child) 
        #TODO this duplication might cause a problem
        count_left_parent[buffer_index] = True
      stack.roots.append(child)
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
        count_left_children[buffer_index] += 1
  return actions, words, labels, features





