# Author: Jan Buys
# Code credit: pytorch example word_language_model

import torch.nn as nn
from torch.autograd import Variable

class RNNEncoder(nn.Module):
    """Container module with an embedding layer and recurrent module."""

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, bias=False)
        #TODO biLSTM

        self.init_weights()

    def init_weights(self): #TODO check 
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size,
                                    self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size,
                                    self.hidden_size).zero_()))

    def forward(self, inp, hidden):
        emb = self.embed(inp)
        output, hidden = self.rnn(emb, hidden)
        return output

