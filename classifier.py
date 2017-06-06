# Author: Jan Buys

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data_utils

class Classifier(nn.Module):
  """Module with classifier for parsing."""

  def __init__(self, num_features, num_indicators, feature_size, hidden_size, 
      output_size, non_lin, use_cuda):
    super(Classifier, self).__init__()
    self.use_cuda = use_cuda
    self.non_lin = non_lin

    self.encode_size = feature_size*num_features
    self.encode = nn.Linear(self.encode_size, hidden_size)
    self.project = nn.Linear(hidden_size, output_size)
    if num_indicators > 0:
      self.indicator_biases = nn.Linear(num_indicators, hidden_size, False)
    else:
      self.indicator_biases = None
    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.encode.bias.data.fill_(0)
    self.encode.weight.data.uniform_(-initrange, initrange)
    #nn.init.xavier_uniform(self.encode.weight)
    self.project.bias.data.fill_(1)
    self.project.weight.data.uniform_(-initrange, initrange)
    if self.indicator_biases is not None:
      self.indicator_biases.weight.data.uniform_(-initrange, initrange)

  def forward(self, features, indicators=None): 
    # features dim: [batch_size, num_features, feature_size] 
    # indicators dim: [batch_size, num_indicators], values [-1, 1]
    flat_features = features.view(-1, self.encode_size)
    hidden = self.encode(flat_features)

    if self.non_lin == data_utils._RELU:
      hidden_nonlin = F.relu(hidden)
    elif self.non_lin == data_utils._TANH:
      hidden_nonlin = F.tanh(hidden)
    elif self.non_lin == data_utils._SIG:
      hidden_nonlin = F.sigmoid(hidden)
    else:
      hidden_nonlin = hidden

    if self.indicator_biases is not None and indicators is not None:
      hidden_nonlin += self.indicator_biases(indicators)

    logits = self.project(hidden_nonlin)
    return logits


