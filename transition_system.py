# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source

import numpy as np

import torch
import torch.nn as nn

import rnn_encoder
import classifier
import binary_classifier

class TransitionSystem():
  def __init__(self, vocab_size, num_relations, num_features, num_transitions,
      embedding_size, hidden_size, num_layers, dropout, init_weight_range, 
      bidirectional, predict_relations, generative, decompose_actions, 
      stack_next, batch_size, use_cuda, model_path, load_model):
    self.use_cuda = use_cuda
    self.vocab_size = vocab_size
    self.decompose_actions = decompose_actions
    self.predict_relations = predict_relations
    self.generative = generative
    self.stack_next = stack_next
    self.num_features = num_features
    self.num_relations = num_relations
    self.num_transitions = num_transitions
    self.log_normalize = nn.LogSoftmax()
    self.binary_normalize = nn.Sigmoid()
    self.feature_size = (hidden_size*2 if bidirectional else hidden_size)

    if load_model:
      assert model_path != ''
      print('Loading models')
      model_fn = model_path + '_encoder.pt'
      with open(model_fn, 'rb') as f:
        self.encoder_model = torch.load(f)
      model_fn = model_path + '_transition.pt'
      with open(model_fn, 'rb') as f:
        self.transition_model = torch.load(f)
      if predict_relations:
        model_fn = model_path + '_relation.pt'
        with open(model_fn, 'rb') as f:
          self.relation_model = torch.load(f)
      else:
        self.relation_model = None
      if generative:
        model_fn = model_path + '_word.pt'
        with open(model_fn, 'rb') as f:
          self.word_model = torch.load(f)
      else:
        self.word_model = None
      if decompose_actions:
        model_fn = model_path + '_direction.pt'
        with open(model_fn, 'rb') as f:
          self.direction_model = torch.load(f)
      else:
        self.direction_model = None

    else: 
      self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, 
          embedding_size, hidden_size, num_layers, dropout, init_weight_range,
          bidirectional=bidirectional, use_cuda=use_cuda)
        
      if decompose_actions:
        self.transition_model = binary_classifier.BinaryClassifier(num_features, 
          self.feature_size, hidden_size, use_cuda) 
        self.direction_model = binary_classifier.BinaryClassifier(num_features, 
          self.feature_size, hidden_size, use_cuda) 
      else:
        self.transition_model = classifier.Classifier(num_features, self.feature_size, 
          hidden_size, num_transitions, use_cuda) 
        self.direction_model = None

      if predict_relations: #TODO use extended feature space
        self.relation_model = classifier.Classifier(num_features, self.feature_size, 
            hidden_size, num_relations, use_cuda)
      else:
        self.relation_model = None
        
      if generative:
        self.word_model = classifier.Classifier(num_features, self.feature_size,
                hidden_size, vocab_size, use_cuda)
      else:
        self.word_model = None

    if use_cuda:
      self.encoder_model.cuda()
      self.transition_model.cuda()
      if predict_relations:
        self.relation_model.cuda()
      if generative:
        self.word_model.cuda()
      if decompose_actions:
        self.direction_model.cuda()

  def store_model(self, path):
    model_fn = path + '_encoder.pt'
    with open(model_fn, 'wb') as f:
      torch.save(self.encoder_model, f)
    model_fn = path + '_transition.pt'
    with open(model_fn, 'wb') as f:
      torch.save(self.transition_model, f)
    if self.predict_relations:
      model_fn = path + '_relation.pt'
      with open(model_fn, 'wb') as f:
        torch.save(self.relation_model, f)
    if self.generative:
      model_fn = path + '_word.pt'
      with open(model_fn, 'wb') as f:
        torch.save(self.word_model, f)
    if self.decompose_actions:
      model_fn = path + '_direction.pt'
      with open(model_fn, 'wb') as f:
        torch.save(self.direction_model, f)

 
