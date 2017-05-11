# Author: Jan Buys
# Code credit: pytorch example word_language_model

import torch.nn as nn
from torch.autograd import Variable

class RNNEncoder(nn.Module):
  """Container module with an embedding layer and recurrent module."""

  def __init__(self, vocab_size, emb_size, hidden_size, num_layers,
    dropout=0.0, bidirectional=False, use_cuda=False):
    super(RNNEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_directions = 2 if bidirectional else 1
    self.use_cuda = use_cuda

    self.drop = nn.Dropout(dropout)
    self.embed = nn.Embedding(vocab_size, emb_size)
    self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, dropout=dropout,
        bias=False, bidirectional=bidirectional) #TODO include extra bias or not?
    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.embed.weight.data.uniform_(-initrange, initrange)
    #nn.init.xavier_uniform(self.embed.weight)

  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    w1 = weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_size).zero_()
    w2 = weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_size).zero_()

    if self.use_cuda:
      return (Variable(w1).cuda(), Variable(w2).cuda())
    else:
      return (Variable(w1), Variable(w2))
    
  def forward(self, inp, hidden): 
    # inp: tuple of (seq length, batch_size)
    # hidden: tuple of (layers, batch_size, hidden_size)
    emb = self.drop(self.embed(inp))
    output, hidden = self.rnn(emb, hidden)
    output = self.drop(output)
    return output

