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
        0, feature_size, hidden_size, non_lin, use_cuda) 
    self.word_model = classifier.Classifier(num_features, 0, feature_size,
         hidden_size, vocab_size, gen_non_lin, False, use_cuda)

    self.log_normalize = nn.LogSoftmax()
    self.binary_log_normalize = nn.LogSigmoid()


  def _inside_algorithm(self, encoder_features, sentence, batch_size):
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

    sh_log_probs_list = self.binary_log_normalize(-self.transition_model(features)).view(num_pairs, batch_size)
    re_rev_log_probs_list = self.binary_log_normalize(self.transition_model(rev_features)).view(num_pairs, batch_size)

    # dim [num_pairs*batch_size, output_size] -> break dim
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
        batch_size).fill_(-np.inf), self.use_cuda)

    word_prob_list = nn_utils.to_var(torch.FloatTensor(table_size, table_size, 
        batch_size).fill_(-np.inf), self.use_cuda)

    # init word probs
    if self.stack_next:
      table[0, 1] = nn_utils.to_var(torch.FloatTensor(batch_size).fill_(0), self.use_cuda)
    else:  
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], 
                                               self.use_cuda)
      # dim [batch_size x vocab_size]
      init_word_distr = self.log_normalize(self.word_model(init_features))
      table[0, 1] = torch.gather(init_word_distr, 1, sentence[1].view(-1, 1))

    # pre-compute word probs
    for i in range(sent_length-1): 
      for j in range(i+1, sent_length):
        index = get_feature_index(i, j)
        word_prob = torch.gather(word_distr_list[index], 1,
            sentence[j if self.stack_next else j+1].view(-1, 1))
        word_prob_list[i, j] = sh_log_probs_list[index] + word_prob

    for j in range(1, sent_length):
      table[j, j+1] = 0

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        re_start_ind = get_rev_feature_index(i+1, j)
        re_end_ind = get_rev_feature_index(j-1, j) + 1
        #start_ind = get_feature_index(i+1, j)
        #end_ind = get_feature_index(j-1, j) + 1
        
        # vectorization 
        sh_probs = word_prob_list[i, i+1:j]
        re_probs = re_rev_log_probs_list[re_start_ind:re_end_ind]
        #shre_block_probs = torch.stack((sh_probs, re_probs), 0)
        #shre_probs = nn_utils.log_sum_exp(shre_block_probs, 0)
        shre_probs = sh_probs + re_probs
        all_block_scores = table[i, i+1:j] + table[i+1:j, j] + shre_probs
        #if i == 0 and j == sent_length:
        #  print(shre_probs.size())
        #  print(table[i, i+1:j].size())
        #  print(all_block_scores)
        table[i, j] = nn_utils.log_sum_exp(all_block_scores, 0)
    
    #print(table[0, sent_length])
    final_score = re_rev_log_probs_list[get_rev_feature_index(0, sent_length)]
    return table[0, sent_length] + final_score


  def _decode_action_sequence(self, encoder_features, word_ids, actions):
    """Execute a given action sequence, also find best relations."""
    # only store indexes on the stack
    stack = []
    buffer_index = 0
    sent_length = len(word_ids) - 1

    buffer_shift_dependents = [-1 for _ in word_ids] # stack top when generated
    stack_shift_dependents = [-1 for _ in word_ids] # interpret SH as (eager) RA
    reduce_dependents = [-1 for _ in word_ids]

    for action in actions:
      s0 = stack[-1] if len(stack) > 0 else 0

      if len(stack) > 1: # allowed to re
        if buffer_index == sent_length:
          assert action == data_utils._SRE
        
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
        
    return actions, stack_shift_dependents


  def _viterbi_algorithm(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    reduce_log_probs = np.zeros([seq_length, seq_length])
    shift_log_probs = np.zeros([seq_length, seq_length])
    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    if self.embed_only_gen:
      gen_features = nn_utils.batch_feature_selection(encoder_features[0], 
          seq_length, self.use_cuda, stack_next=self.stack_next)
      word_distr_list = self.log_normalize(self.word_model(gen_features))
    else:        
      word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        reduce_log_probs[i, j] = re_log_probs_list[counter, 0]
        shift_log_probs[i, j] = sh_log_probs_list[counter, 0]
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
        score = shift_log_probs[i, j-1]
        if self.word_model is not None:
          score += word_log_probs[i, j-1]
        table[i, j-1, j] = score
      for i in range(j-2, -1, -1):
        for l in range(max(i, 1)): # l=0 if i=0
          block_scores = [] # adjust indexes after collecting scores
          block_directions = []
          for k in range(i+1, j):
            score = table[l, i, k] + table[i, k, j] + reduce_log_probs[k, j]
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

    loss = -torch.sum(self._inside_algorithm(
        encoder_features, sentence, batch_size))
    return loss

  def forward(self, sentence):
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    return self._viterbi_algorithm(encoder_features, word_ids)



