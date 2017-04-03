# Author: Jan Buys

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Classifier(nn.Module):
  """Module with classifier for parsing."""

  def __init__(self, num_features, feature_size, hidden_size, output_size,
      use_cuda):
    super(Classifier, self).__init__()
    self.use_cuda = use_cuda

    self.encode_size = feature_size*num_features
    self.encode = nn.Linear(self.encode_size, hidden_size)
    self.project = nn.Linear(hidden_size, output_size)

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.encode.bias.data.fill_(0)
    self.encode.weight.data.uniform_(-initrange, initrange)
    #nn.init.xavier_uniform(self.encode.weight)
    self.project.bias.data.fill_(1)
    self.project.weight.data.uniform_(-initrange, initrange)

  def forward(self, features): # features dim: [batch_size, num_features, feature_size]
    flat_features = features.view(-1, self.encode_size)
    hidden = self.encode(flat_features)
    hidden_nonlin = F.tanh(hidden)
    logits = self.project(hidden_nonlin)
    return logits


