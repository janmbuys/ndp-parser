# Author: Jan Buys
# Code credit: pytorch example word_language_model

import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an embedding layer, recurrent module, and
    projection layer."""

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, bias=False)
        self.project = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self): #TODO check 
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.project.bias.data.fill_(0)
        self.project.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size,
                                    self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size,
                                    self.hidden_size).zero_()))

    def forward(self, inp, hidden):
        emb = self.embed(inp)
        output, hidden = self.rnn(emb, hidden)
        # Reshape to apply projection layer
        output_flatten = output.view(output.size(0)*output.size(1), 
                                     output.size(2))
        logits_flatten = self.project(output_flatten)
        logits = logits_flatten.view(output.size(0), output.size(1), 
                                     logits_flatten.size(1))
        return logits, hidden


