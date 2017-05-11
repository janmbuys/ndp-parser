# Author: Jan Buys
# Code credit: pytorch example word_language_model

import torch.nn as nn
from torch.autograd import Variable

class RNNLM(nn.Module):
    """Container module with an embedding layer, recurrent module, and
    projection layer."""

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers,
        dropout=0.0, init_weight_range=0.1, xavier_init=False, tie_weights=False, use_cuda=False):
        super(RNNLM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_cuda = use_cuda

        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, dropout=dropout,
            bias=False)
        self.project = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
          assert emb_size == hidden_size, 'Embedding and hidden dimensions must be equal for tied weights.'
          self.project.weight = self.embed.weight

        self.init_weights(init_weight_range, xavier_init)

    def init_weights(self, initrange=0.1, xavier_init=False):
        if xavier_init:
          nn.init.xavier_uniform(self.embed.weight)
          nn.init.xavier_uniform(self.project.weight)
          for k in range(self.num_layers):
            nn.init.xavier_uniform(self.rnn.weight_ih_l)
            nn.init.xavier_uniform(self.rnn.weight_hh_l)
            self.rnn.bias_ih_l.fill_(0)
            self.rnn.bias_hh_l.fill_(0)
        else:  
          self.embed.weight.data.uniform_(-initrange, initrange)
          self.project.weight.data.uniform_(-initrange, initrange)
        self.project.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        w1 = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        w2 = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()

        if self.use_cuda:
          return (Variable(w1).cuda(), Variable(w2).cuda())
        else:
          return (Variable(w1), Variable(w2))

    def forward(self, inp, hidden):
        # inp: tuple of (seq length, batch_size)
        # hidden: tuple of (layers, batch_size, hidden_size)
        # logits: (seq_length, batch_size, vocab_size)
        emb = self.drop(self.embed(inp))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        # Reshape to apply projection layer
        output_flatten = output.view(output.size(0)*output.size(1), 
                                     output.size(2))
        logits_flatten = self.project(output_flatten)
        logits = logits_flatten.view(output.size(0), output.size(1), 
                                     logits_flatten.size(1))
        return logits, hidden


