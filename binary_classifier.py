# Author: Jan Buys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data_utils

class BinaryClassifier(nn.Module):
  """Module with binary classifier for parsing."""

  def __init__(self, num_features, num_indicators, feature_size, hidden_size,
      non_lin, use_cuda):
    super(BinaryClassifier, self).__init__()
    self.use_cuda = use_cuda
    self.non_lin = non_lin
    self.num_features = num_features
    self.feature_size = feature_size
    assert num_indicators >= 0 and num_indicators <= 2
    self.num_indicators = num_indicators

    self.encode_size = feature_size*(num_features + num_indicators)
    self.encode = nn.Linear(self.encode_size, hidden_size)
    self.project = nn.Linear(hidden_size, 1)

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.encode.bias.data.fill_(0)
    self.encode.weight.data.uniform_(-initrange, initrange)
    #nn.init.xavier_uniform(self.encode.weight)
    self.project.bias.data.fill_(1)
    self.project.weight.data.uniform_(-initrange, initrange)

  def forward(self, features, indicator_index=None): 
    # features dim: [batch_size, num_features, feature_size]
    if self.num_indicators > 0:
      assert indicator_index is not None
      col_features = features.view(-1, self.num_features, self.feature_size)
      zers = nn_utils.to_var(torch.zeros(col_features.size()[0], 
          self.num_indicators, col_features.size()[2]), self.use_cuda)
      cat_features = torch.cat((col_features, zers), 1)
      positions = data_utils.indicators_to_positions(indicator_index, 
          self.num_indicators)
      positions_var = nn_utils.to_var(torch.LongTensor(positions), self.use_cuda)
      flat_features = torch.index_select(cat_features, 1, positions_var).view(-1,
          self.encode_size)
    else:
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

    logits = self.project(hidden_nonlin)
    return logits


