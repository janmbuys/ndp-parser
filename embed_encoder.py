# Author: Jan Buys
# Code credit: pytorch example word_language_model

import torch.nn as nn
from torch.autograd import Variable

class EmbedEncoder(nn.Module):
  """Container module with an embedding layer and recurrent module."""

  def __init__(self, vocab_size, emb_size, hidden_size,
    dropout=0.0, init_weight_range=0.1, use_cuda=False):
    super(EmbedEncoder, self).__init__()
    assert emb_size == hidden_size
    self.use_cuda = use_cuda
    self.hidden_size = hidden_size

    self.drop = nn.Dropout(dropout)
    self.embed = nn.Embedding(vocab_size, emb_size)
    self.init_weights(init_weight_range)

  def init_weights(self, initrange=0.1):
    self.embed.weight.data.uniform_(-initrange, initrange)

  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data #TODO what is this?
    w = weight.new(1, batch_size, self.hidden_size).zero_()

    if self.use_cuda:
      return Variable(w).cuda()
    else:
      return Variable(w)

  def forward(self, inp, hidden): 
    # inp: tuple of (seq length, batch_size)
    # hidden: tuple of (layers, batch_size, hidden_size)
    emb = self.drop(self.embed(inp))
    return emb

