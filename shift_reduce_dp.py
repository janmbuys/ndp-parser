# Author: Jan Buys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import classifier
import binary_classifier
import rnn_encoder
import embed_encoder
import nn_utils
import data_utils

class ShiftReduceDP(nn.Module):
  """Stack-based generative model with dynamic programming inference.""" 

  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,
               dropout, init_weight_range, non_lin, gen_non_lin,
               stack_next, embed_only, embed_only_gen, use_cuda):
    super(ShiftReduceDP, self).__init__()
    self.use_cuda = use_cuda
    self.stack_next = stack_next
    self.embed_only_gen = embed_only_gen
    num_features = 2

    if embed_only:
      self.encoder_model = embed_encoder.EmbedEncoder(vocab_size, embedding_size, 
          hidden_size, dropout, init_weight_range, use_cuda=use_cuda)
    else:
      self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, embedding_size, 
          hidden_size, num_layers, dropout, init_weight_range, bidirectional=False, use_cuda=use_cuda)

    feature_size = hidden_size
    self.transition_model = binary_classifier.BinaryClassifier(num_features, 
        feature_size, hidden_size, non_lin, use_cuda) 
    self.word_model = classifier.Classifier(num_features, 0, feature_size,
         hidden_size, vocab_size, gen_non_lin, use_cuda)

    self.log_normalize = nn.LogSoftmax()
    self.binary_normalize = nn.Sigmoid()


  def _inside_algorithm_iterative_old(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda)
    re_probs_list = self.binary_normalize(self.transition_model(features)).view(-1)
    word_distr_list = self.log_normalize(self.word_model(features))

    init_features = nn_utils.select_features(encoder_features[1], [0, 0], 
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
 
    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        for l in range(max(i, 1)):
          block_scores = []
          score = None
          for k in range(i+1, j):
            re_score = torch.log(re_probs_list[get_feature_index(k, j)])
            t_score = table[l][i][k] + table[i][k][j] + re_score
            block_scores.append(t_score)
          table[l][i][j] = nn_utils.log_sum_exp_1d(torch.cat(block_scores).view(1, -1))
    return table[0][0][sent_length]

  def _inside_algorithm_iterative(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)
    max_dependency_length = 20

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_probs_list = self.binary_normalize(self.transition_model(features)).view(-1)
    word_distr_list = self.log_normalize(self.word_model(features))

    init_features = nn_utils.select_features(encoder_features[1], [0, 0], 
                                             self.use_cuda)
    init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
    # do simple arithmetic to access distribution list entries
    def get_feature_index(i, j):
      return int((2*seq_length-i-1)*(i/2) + j-i-1)

    table_size = len(word_ids)
    table = nn_utils.to_var(torch.FloatTensor(table_size, table_size, 
        table_size).fill_(-np.inf), self.use_cuda)

    # word probs
    table[0, 0, 1] = init_word_distr[word_ids[1]]
    for i in range(sent_length-1): # features goes 1 index further
      #for j in range(i+1, sent_length): # features goes 1 index further
      for j in range(i+1, min(sent_length, i + max_dependency_length)):
        index = get_feature_index(i, j)
        table[i, j, j+1] = (torch.log1p(-re_probs_list[index]) 
            + word_distr_list[index, word_ids[j if self.stack_next else j+1]])
 
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
          table[l, i, j] = nn_utils.log_sum_exp_1d(torch.cat(block_scores).view(1, -1))
    return table[0, 0, sent_length]

  def _inside_algorithm_iterative_vectorized(self, encoder_features, sentence,
      batch_size):
    # enc feature dim length x batch x state_size
    # sentence dim length x batch
    sent_length = sentence.size()[0] - 1
    seq_length = sentence.size()[0]

    # dim [num_pairs, batch_size, 2, state_size]
    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    rev_features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, rev=True, stack_next=self.stack_next)
    num_pairs = features.size()[0]

    eps = nn_utils.to_var(torch.FloatTensor(num_pairs, batch_size).fill_(
        np.exp(-10)), self.use_cuda)
    # dim [num_pairs*batch_size, output_size] -> break dim
    re_probs_list = self.binary_normalize(self.transition_model(features)).view(num_pairs, batch_size)
    sh_log_probs_list = torch.log1p(-re_probs_list+eps)
    re_rev_probs_list = self.binary_normalize(self.transition_model(rev_features)).view(num_pairs, batch_size)
    re_rev_log_probs_list = torch.log(re_rev_probs_list+eps)

    if self.embed_only_gen:
      gen_features = nn_utils.batch_feature_selection(encoder_features[0], 
          seq_length, self.use_cuda, stack_next=self.stack_next)
      word_distr_list = self.log_normalize(self.word_model(gen_features)).view(
          num_pairs, batch_size, -1) 
    else:
      word_distr_list = self.log_normalize(self.word_model(features)).view(
          num_pairs, batch_size, -1) 

    # enumerate indexes
    inds_table = np.zeros((seq_length, seq_length), dtype=np.int)
    counter = 0
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        inds_table[i, j-1 if self.stack_next else j] = counter
        counter += 1

    def get_feature_index(i, j):
      return inds_table[i, j-1 if self.stack_next else j]

    # rather enumerate indexes for the reverse
    rev_inds_table = np.zeros((seq_length, seq_length), dtype=np.int)
    counter = 0
    for j in range(1, seq_length):
      for i in range(j):
        rev_inds_table[i, j-1 if self.stack_next else j] = counter
        counter += 1

    def get_rev_feature_index(i, j):
      return rev_inds_table[i, j-1 if self.stack_next else j]

    table_size = sentence.size()[0]
    table = nn_utils.to_var(torch.FloatTensor(table_size, table_size, 
        table_size, batch_size).fill_(-np.inf), self.use_cuda)

    # word probs
    if self.stack_next:
      table[0, 0, 1] = nn_utils.to_var(torch.FloatTensor(batch_size).fill_(0), self.use_cuda)
    else:  
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], 
                                               self.use_cuda)
      # dim [batch_size x vocab_size]
      init_word_distr = self.log_normalize(self.word_model(init_features))
      table[0, 0, 1] = torch.gather(init_word_distr, 1, sentence[1].view(-1, 1))
   
    for i in range(sent_length-1): 
      for j in range(i+1, sent_length):
        index = get_feature_index(i, j)
        word_prob = torch.gather(word_distr_list[index], 1,
            sentence[j if self.stack_next else j+1].view(-1, 1))
        table[i, j, j+1] = sh_log_probs_list[index] + word_prob

    if False: # More complicated, not faster
      for i in range(sent_length-1): 
        start_index = get_feature_index(i, i+1)
        end_index = get_feature_index(i, sent_length-1) + 1
        
        word_probs = nn_utils.to_var(torch.FloatTensor(sent_length - (i + 1),
            batch_size).fill_(-np.inf), self.use_cuda)

        for j in range(i+1, sent_length):
          index = get_feature_index(i, j)
          word_probs[j - (i + 1)] = torch.gather(word_distr_list[index], 1,
              sentence[j if self.stack_next else j+1].view(-1, 1))
        sh_probs = sh_log_probs_list[start_index:end_index] + word_probs

        # Cannot do scatter asign to table directly
        word_table = nn_utils.to_var(torch.FloatTensor(sent_length - (i+1), 
            sent_length - i, batch_size).fill_(-np.inf), self.use_cuda)

        # Indexing for scatter asignment.
        range_var = nn_utils.to_var(torch.LongTensor(range(1, 
            sent_length-i)).view(-1, 1).repeat(1, batch_size), self.use_cuda)
        
        word_table.scatter_(1, range_var.view(-1, 1, batch_size), 
            sh_probs.view(-1, 1, batch_size))
        table[i, i+1:sent_length, i+1:sent_length+1] = word_table

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        start_ind = get_rev_feature_index(i+1, j)
        end_ind = get_rev_feature_index(j-1, j) + 1
              
        # Super vectorization 
        re_probs = re_rev_log_probs_list[start_ind:end_ind]
        temp_right = table[i, i+1:j, j] + re_probs.view(1, -1, batch_size)
        all_block_scores = (table[0:max(i, 1), i, i+1:j] 
                            + temp_right.expand(max(i, 1), gap - 1, batch_size))
        table[0:max(i, 1), i, j] = nn_utils.log_sum_exp(all_block_scores, 1)
     
    final_score = re_rev_log_probs_list[get_rev_feature_index(0, sent_length)]
    return table[0, 0, sent_length] + final_score

  def _decode_action_sequence(self, encoder_features, word_ids, actions):
    """Execute a given action sequence, also find best relations."""
    # only store indexes on the stack
    stack = []
    buffer_index = 0
    sent_length = len(word_ids) - 1

    transition_logits = []
    buffer_shift_dependents = [-1 for _ in word_ids] # stack top when generated
    stack_shift_dependents = [-1 for _ in word_ids] # interpret SH as (eager) RA
    reduce_dependents = [-1 for _ in word_ids]

    for action in actions:
      transition_logit = None

      s0 = stack[-1] if len(stack) > 0 else 0

      if len(stack) > 1: # allowed to re
        position = nn_utils.extract_feature_positions(buffer_index, s0,
            stack_next=self.stack_next)
        features = nn_utils.select_features(encoder_features[1], position, self.use_cuda)
        
        transition_logit = self.transition_model(features)
        if buffer_index == sent_length:
          assert action == data_utils._SRE
        
      transition_logits.append(transition_logit)

      # excecute action
      if action == data_utils._SSH:
        if len(stack) > 0:
          stack_shift_dependents[buffer_index] = stack[-1]
          if buffer_index + 1 < len(word_ids):
            buffer_shift_dependents[buffer_index+1] = stack[-1]
        stack.append(buffer_index) 
        buffer_index += 1
      else:  
        assert len(stack) > 0
        child = stack.pop()
        reduce_dependents[child] = buffer_index
        
    return transition_logits, actions, stack_shift_dependents


  def _viterbi_algorithm(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)
    eps = np.exp(-10) # used to avoid division by 0

    # compute all sh/re and word probabilities
    reduce_probs = np.zeros([seq_length, seq_length])
    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_probs_list = nn_utils.to_numpy(self.binary_normalize(self.transition_model(features)))
    if self.embed_only_gen:
      gen_features = nn_utils.batch_feature_selection(encoder_features[0], 
          seq_length, self.use_cuda, stack_next=self.stack_next)
      word_distr_list = self.log_normalize(self.word_model(gen_features))
    else:        
      word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        reduce_probs[i, j] = re_probs_list[counter, 0]
        if j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = len(word_ids)
    table = np.empty([table_size, table_size, table_size])
    table.fill(-np.inf) # log probabilities
    split_indexes = np.zeros((table_size, table_size, table_size), dtype=np.int)

    # first word prob 
    if self.stack_next:
      table[0, 0, 1] = 0
    else:
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], 
                                               self.use_cuda)
      init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
      table[0, 0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])

    for j in range(2, sent_length+1):
      for i in range(j-1):
        score = np.log(1 - reduce_probs[i, j-1] + eps) 
        if self.word_model is not None:
          score += word_log_probs[i, j-1]
        table[i, j-1, j] = score
      for i in range(j-2, -1, -1):
        for l in range(max(i, 1)): # l=0 if i=0
          block_scores = [] # adjust indexes after collecting scores
          block_directions = []
          for k in range(i+1, j):
            score = table[l, i, k] + table[i, k, j] + np.log(reduce_probs[k, j] + eps)
            block_scores.append(score)
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
    batch_size = sentence.size()[1]
    encoder_state = self.encoder_model.init_hidden(batch_size)
    encoder_features = self.encoder_model(sentence, encoder_state)

    loss = -torch.sum(self._inside_algorithm_iterative_vectorized(
        encoder_features, sentence, batch_size))
    return loss

  def forward(self, sentence):
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    return self._viterbi_algorithm(encoder_features, word_ids)



