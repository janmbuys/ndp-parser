# Author: Jan Buys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import classifier
import binary_classifier
import rnn_encoder
import nn_utils
import data_utils

class DPStack(nn.Module):
  """Stack-based generative model with dynamic programming inference.""" 

  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,
               dropout, num_features, use_cuda):
    super(DPStack, self).__init__()
    self.use_cuda = use_cuda
    self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, embedding_size, 
        hidden_size, num_layers, dropout, False, use_cuda)

    feature_size = hidden_size
    self.transition_model = binary_classifier.BinaryClassifier(num_features, 
        feature_size, hidden_size, use_cuda) 
    self.word_model = classifier.Classifier(num_features, feature_size,
         hidden_size, vocab_size, use_cuda)

    self.log_normalize = nn.LogSoftmax()
    self.binary_normalize = nn.Sigmoid()

  def _inside_algorithm_iterative_old(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    features = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda)
    re_probs_list = self.binary_normalize(self.transition_model(features)).view(-1)
    word_distr_list = self.log_normalize(self.word_model(features))

    init_features = nn_utils.select_features(encoder_features, [0, 0], 
                                             self.use_cuda)
    init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
    # do simple arithmetic to access distribution list entries
    def get_feature_index(i, j):
      return int((2*seq_length-i-1)*(i/2) + j-i-1)

    table_size = len(word_ids)
    table = []
    for _ in range(table_size):
      in_table = []
      for _ in range(table_size):
        in_table.append([None for _ in range(table_size)])
      table.append(in_table)

    # word probs
    table[0][0][1] = init_word_distr[word_ids[1]]
    for i in range(sent_length-1): # features goes 1 index further
      for j in range(i+1, sent_length): # features goes 1 index further
        index = get_feature_index(i, j)
        table[i][j][j+1] = (torch.log1p(-re_probs_list[index]) 
                            + word_distr_list[index, word_ids[j+1]])
 
    for gap in range(2, sent_length+1): #TODO this is not the old one
      #print(gap)
      for i in range(sent_length+1-gap):
        j = i + gap
        for l in range(max(i, 1)):
          block_scores = []
          score = None
          for k in range(i+1, j):
            re_score = torch.log(re_probs_list[get_feature_index(k, j)])
            t_score = table[l][i][k] + table[i][k][j] + re_score
            #score = score + t_score if score is not None else t_score
            block_scores.append(t_score)
          #table[l][i][j] = score
          table[l][i][j] = nn_utils.log_sum_exp(torch.cat(block_scores).view(1, -1))
    return table[0][0][sent_length]

  def _inside_algorithm_iterative(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)
    max_dependency_length = 20

    features = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda)
    re_probs_list = self.binary_normalize(self.transition_model(features)).view(-1)
    word_distr_list = self.log_normalize(self.word_model(features))

    init_features = nn_utils.select_features(encoder_features, [0, 0], 
                                             self.use_cuda)
    init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
    # do simple arithmetic to access distribution list entries
    def get_feature_index(i, j):
      return int((2*seq_length-i-1)*(i/2) + j-i-1)

    table_size = len(word_ids)
    table_ts = torch.FloatTensor(table_size, table_size, table_size).fill_(-np.inf)
    if self.use_cuda:
      table = Variable(table_ts).cuda()
    else:
      table = Variable(table_ts)

    # word probs
    table[0, 0, 1] = init_word_distr[word_ids[1]]
    for i in range(sent_length-1): # features goes 1 index further
      #for j in range(i+1, sent_length): # features goes 1 index further
      for j in range(i+1, min(sent_length, i + max_dependency_length)):
        index = get_feature_index(i, j)
        table[i, j, j+1] = (torch.log1p(-re_probs_list[index]) 
                            + word_distr_list[index, word_ids[j+1]])
 
    for gap in range(2, sent_length+1):
      #print(gap)
      for i in range(sent_length+1-gap):
        j = i + gap
        #for l in range(max(i, 1)):
        for l in range(max(0, i-max_dependency_length), max(i, 1)):
          block_scores = []
          score = None
          for k in range(max(i+1, j-max_dependency_length),  j):
          #for k in range(i+1, j):
            re_score = torch.log(re_probs_list[get_feature_index(k, j)])
            t_score = table[l, i, k] + table[i, k, j] + re_score
            #score = score + t_score if score is not None else t_score
            block_scores.append(t_score)
          #table[l][i][j] = score
          table[l, i, j] = nn_utils.log_sum_exp(torch.cat(block_scores).view(1, -1))
    return table[0, 0, sent_length]

  def _inside_algorithm_iterative_vectorized(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    features = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda)
    features_rev = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda, rev=True)

    re_probs_list = self.binary_normalize(self.transition_model(features)).view(-1)
    re_probs_list_rev = self.binary_normalize(self.transition_model(features_rev)).view(-1)

    word_distr_list = self.log_normalize(self.word_model(features))

    init_features = nn_utils.select_features(encoder_features, [0, 0], 
                                             self.use_cuda)
    init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
    # do simple arithmetic to access distribution list entries
    def get_feature_index(i, j):
      return int((2*seq_length-i-1)*(i/2) + j-i-1)

    rev_inds_table = np.zeros((seq_length, seq_length), dtype=np.int)
    counter = 0
    for j in range(1, seq_length):
      for i in range(j):
        rev_inds_table[i,j] = counter
        counter += 1

    def get_rev_feature_index(i, j):
      return rev_inds_table[i, j]

    table_size = len(word_ids)
    table_ts = torch.FloatTensor(table_size, table_size, table_size).fill_(-np.inf)
    if self.use_cuda:
      table = Variable(table_ts).cuda()
    else:
      table = Variable(table_ts)

    # word probs
    table[0, 0, 1] = init_word_distr[word_ids[1]]
    for i in range(sent_length-1): # features goes 1 index further
      start_index = get_feature_index(i, i+1)
      end_index = get_feature_index(i, sent_length-1) # last j+1

      word_probs_ts = torch.FloatTensor(sent_length - (i + 1)).fill_(-np.inf)
      if self.use_cuda:
        word_probs = Variable(word_probs_ts).cuda()
      else:
        word_probs = Variable(word_probs_ts)

      # word_distr_list[start_index:end_index, *]
      for j in range(i+1, sent_length): # features goes 1 index further
        index = get_feature_index(i, j)
        word_probs[j - (i + 1)] = word_distr_list[index, word_ids[j+1]]
      re_probs = torch.log1p(-re_probs_list[start_index:end_index+1])

      # Note that the diagonal contains 0's.
      table[i, i+1:table_size, i+1:table_size] = torch.diag(re_probs + word_probs, 1)
      
      #for j in range(i+1, sent_length): # features goes 1 index further
      #  index = get_feature_index(i, j)
      #  table[i, j, j+1] = (torch.log1p(-re_probs_list[index]) 
      #                      + word_distr_list[index, word_ids[j+1]])
 
    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        start_ind = get_rev_feature_index(i+1, j)
        end_ind = get_rev_feature_index(j-1, j) + 1
        re_temp = torch.log(re_probs_list[start_ind:end_ind].view(1, -1))
      
        # This vectorization actually gives an order of magnitude speedup!
        all_block_scores = (table[0:max(i, 1), i, i+1:j]
                            + table[i, i+1:j, j].expand(max(i, 1), j-i-1)
                            + re_temp.expand(max(i, 1), j-i-1))
        table[0:max(i, 1), i, j] = nn_utils.log_sum_exp_2d(all_block_scores)
          
        #for l in range(max(i, 1)):
        #  block_scores = table[l, i, i+1:j] + temp 
        #  table[l, i, j] = nn_utils.log_sum_exp(block_scores.view(1, -1))

    return table[0, 0, sent_length]



  def _inside_algorithm_recusive(self,encoder_features, word_ids): 
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    features = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda)
    re_probs_list = self.binary_normalize(self.transition_model(features))
    word_distr_list = self.log_normalize(self.word_model(features))

    init_features = nn_utils.select_features(encoder_features, [0, 0], 
                                             self.use_cuda)
    init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))

    # do simple arithmetic to access distribution list entries
    def get_feature_index(i, j):
      return int((2*seq_length-i-1)*(i/2) + j-i-1)

    def inside_op(l, i, j):
      if l == 0 and i == 0 and j == 1:
        return init_word_distr[word_ids[1]]
      if i == j - 1: # word emmision
        index = get_feature_index(l, i)
        return (torch.log1p(-re_probs_list[index, 0]) 
                + word_distr_list[index, word_ids[j]])
      else:
        block_scores = []
        for k in range(i+1, j):
          score = (inside_op(l, i, k) + inside_op(i, k, j) +   
                   torch.log(re_probs_list[get_feature_index(k, j), 0]))
          block_scores.append(score) #TODO rather add one at a time
        return nn_utils.log_sum_exp(torch.cat(block_scores).view(1, -1))

    return inside_op(0, 0, sent_length)


  def _decode_action_sequence(self, encoder_features, word_ids, actions):
    """Execute a given action sequence, also find best relations."""
    # only store indexes on the stack
    stack = []
    buffer_index = 0
    sent_length = len(word_ids) - 1

    transition_logits = []
    shift_dependents = [-1 for _ in word_ids] # stack top when shifted
      # but model actually uses stack top when generated to buffer
    reduce_dependents = [-1 for _ in word_ids] # buffer entry when reduced

    for action in actions:
      transition_logit = None

      s0 = stack[-1] if len(stack) > 0 else 0

      if len(stack) > 1: # allowed to re
        position = nn_utils.extract_feature_positions(buffer_index, s0)
        features = nn_utils.select_features(encoder_features, position, self.use_cuda)
        
        transition_logit = self.transition_model(features)
        if buffer_index == sent_length:
          assert action == data_utils._SRE
        
      transition_logits.append(transition_logit)

      # excecute action
      if action == data_utils._SSH:
        if len(stack) > 0:
          shift_dependents[buffer_index] = stack[-1]
        stack.append(buffer_index) 
        buffer_index += 1
      else:  
        assert len(stack) > 0
        child = stack.pop()
        reduce_dependents[child] = buffer_index
        
    return transition_logits, actions, shift_dependents, reduce_dependents


  def _viterbi_algorithm(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    features = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda)
    re_probs_list = self.binary_normalize(self.transition_model(features)).view(-1)
    word_distr_list = self.log_normalize(self.word_model(features))


    init_features = nn_utils.select_features(encoder_features, [0, 0], 
                                             self.use_cuda)
    init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
    # do simple arithmetic to access distribution list entries
    def get_feature_index(i, j):
      return int((2*seq_length-i-1)*(i/2) + j-i-1)

    table_size = len(word_ids)
    table_ts = torch.FloatTensor(table_size, table_size, table_size).fill_(-np.inf)
    if self.use_cuda:
      table = Variable(table_ts).cuda()
    else:
      table = Variable(table_ts)
    split_indexes = np.zeros((table_size, table_size, table_size), dtype=np.int)
    # word probs
    table[0, 0, 1] = init_word_distr[word_ids[1]]
    for i in range(sent_length-1): # features goes 1 index further
      for j in range(i+1, sent_length): # features goes 1 index further
        index = get_feature_index(i, j)
        table[i, j, j+1] = (torch.log1p(-re_probs_list[index]) 
                            + word_distr_list[index, word_ids[j+1]])
 
    for gap in range(2, sent_length+1):
      #print(gap)
      for i in range(sent_length+1-gap):
        j = i + gap
        for l in range(max(i, 1)):
          block_scores = []
          score = None
          for k in range(i+1, j):
            re_score = torch.log(re_probs_list[get_feature_index(k, j)])
            t_score = table[l, i, k] + table[i, k, j] + re_score
            #score = score + t_score if score is not None else t_score
            block_scores.append(t_score)
          #table[l][i][j] = score
          ind = np.argmax(block_scores)
          k = ind + i + 1
          table[l, i, j] = block_scores[ind]
          split_indexes[l, i, j] = k
 
    def backtrack_path(l, i, j):
      """ Find action sequence for best path. """
      if i == j - 1:
        return [data_utils._SSH]
      else:
        k = split_indexes[l, i, j]
        return (backtrack_path(l, i, k) + backtrack_path(i, k, j) 
                + [data_utils._SRE])

    actions = backtrack_path(0, 0, sent_length)
    return self._decode_action_sequence(encoder_features, word_ids, actions)
  

  def neg_log_likelihood(self, sentence):
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    #return -self._inside_algorithm_iterative(encoder_features, word_ids)
    return -self._inside_algorithm_iterative_vectorized(encoder_features, word_ids)


  def forward(self, sentence):
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    return self._viterbi_algorithm(encoder_features, word_ids)



