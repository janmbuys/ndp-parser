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

class TransitionSystem():
  def __init__(self, vocab_size, num_relations, num_features, num_transitions,
      embedding_size, hidden_size, num_layers, dropout, bidirectional, 
      more_context, batch_size, use_cuda):
    self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, 
        embedding_size, hidden_size, num_layers, dropout,
        bidirectional, use_cuda)

    feature_size = (args.hidden_size*2 if args.bidirectional 
                    else args.hidden_size)
    if args.decompose_actions:
      self.transition_model = binary_classifier.BinaryClassifier(num_features, 
        feature_size, hidden_size, use_cuda) 
      self.direction_model = binary_classifier.BinaryClassifier(num_features, 
        feature_size, hidden_size, use_cuda) 
    else:
      self.transition_model = classifier.Classifier(num_features, feature_size, 
        hidden_size, num_transitions, use_cuda) 
    if args.predict_relations:
      self.relation_model = classifier.Classifier(num_features, feature_size, 
          hidden_size, num_relations, use_cuda)
    else:
      self.relation_model = None
    if args.generative:
      self.word_model = classifier.Classifier(num_features, feature_size,
              hidden_size, vocab_size, use_cuda)

    if use_cuda:
      self.encoder_model.cuda()
      self.transition_model.cuda()
      if args.predict_relations:
        self.relation_model.cuda()
      if args.generative:
        self.word_model.cuda()
      if args.decompose_actions:
        self.direction_model.cuda()




