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

class ArcEagerDP(nn.Module):
  """arc-eager generative model with dynamic programming inference.""" 

  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,
               dropout, init_weight_range, num_features, non_lin, gen_non_lin,
               stack_next, embed_only, use_cuda):
    super(ArcEagerDP, self).__init__()
    self.use_cuda = use_cuda
    self.stack_next = stack_next
    self.generate_actions = [data_utils._SH, data_utils._RA]
    self.num_transitions = 3

    if embed_only:
      self.encoder_model = embed_encoder.EmbedEncoder(vocab_size, embedding_size, 
          hidden_size, dropout, init_weight_range, use_cuda=use_cuda)
    else:
      self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, embedding_size, 
          hidden_size, num_layers, dropout, init_weight_range, bidirectional=False, use_cuda=use_cuda)

    feature_size = hidden_size
    # TODO for now don't decompose actions
    self.transition_model = classifier.Classifier(num_features, 0, 
        feature_size, hidden_size, self.num_transitions, non_lin, use_cuda) 

    self.word_model = classifier.Classifier(num_features, 0, feature_size,
         hidden_size, vocab_size, gen_non_lin, use_cuda)

    self.log_normalize = nn.LogSoftmax()
    #self.normalize = nn.Softmax()
    #self.binary_normalize = nn.Sigmoid()


  def _inside_algorithm_iterative(self, encoder_features, sentence, batch_size):
    assert batch_size == 1, "Batch processing not supported here"
    sent_length = sentence.size()[0] - 1
    seq_length = sentence.size()[0]
    eps = np.exp(-10) # used to avoid division by 0

    # batch feature computation
    enc_features = nn_utils.batch_feature_selection(encoder_features, seq_length, 
        self.use_cuda, stack_next=self.stack_next)
    features = enc_features
    num_items = enc_features.size()[0]

    tr_log_probs_list = self.log_normalize(self.transition_model(features)).view(
        num_items, self.num_transitions)
    word_distr_list = self.log_normalize(self.word_model(features)).view(
        num_items, -1)

    init_features = nn_utils.select_features(encoder_features, [0, 0], self.use_cuda)
    init_word_dist = self.log_normalize(self.word_model(init_features)).view(-1)
    # enumerate indexes
    inds_table = np.zeros((seq_length, seq_length, 2), dtype=np.int)
    counter = 0
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        for c in range(2):
          inds_table[i, j-1 if self.stack_next else j, c] = counter
        counter += 1 # c's use same features for now

    def get_feature_index(i, j, c):
      return inds_table[i, j-1 if self.stack_next else j, c]

    table_size = sentence.size()[0]
    table = nn_utils.to_var(torch.FloatTensor(table_size, 2, table_size, 2,
        table_size).fill_(-np.inf), self.use_cuda)
   
    # word probs
    table[0, 0, 0, 0, 1] = torch.gather(init_word_dist, 0, sentence[1].view(1))
     
    for i in range(sent_length-1):
      for j in range(i+1, sent_length):
        for c in range(2):
          index = get_feature_index(i, j, c)
          word_prob = torch.gather(word_distr_list[index], 0,
              sentence[j if self.stack_next else j+1].view(1))

          table[i, c, j, 0, j+1] = (tr_log_probs_list[index, data_utils._ESH]
              + word_prob)
          table[i, c, j, 1, j+1] = (tr_log_probs_list[index, data_utils._ERA]
              + word_prob)

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        for c in range(1 if i == 0 else 2): # don't vectorize
          temp_right = nn_utils.to_var(torch.FloatTensor(j-i-1), self.cuda)
          for k in range(i+1, j):
            score = nn_utils.to_var(torch.FloatTensor(2), self.cuda)
            score[0] = (table[i, c, k, 0, j] 
                + tr_log_probs_list[get_feature_index(k, j, 0), data_utils._ERE])
            score[1] = (table[i, c, k, 1, j] 
                + tr_log_probs_list[get_feature_index(k, j, 1), data_utils._ERE])

            if j == sent_length:
              temp_right[k-i-1] = score[1]
            else:
              temp_right[k-i-1] = nn_utils.log_sum_exp(score, 0)

          for l in range(max(i, 1)):
            for b in range(1 if i == 0 else 2):
              block_scores = nn_utils.to_var(torch.FloatTensor(j-i-1), self.cuda)
              for k in range(i+1, j):
                block_scores[k-i-1] = table[l, b, i, c, k] + temp_right[k-i-1]
              table[l, b, i, c, j] = nn_utils.log_sum_exp(block_scores, 0)
    return table[0, 0, 0, 0, sent_length] 


  def _decode_action_sequence(self, encoder_features, word_ids, actions):
    stack = []
    stack_has_parent = []
    buffer_index = 0
    sent_length = len(word_ids) - 1

    transition_logits = []
    #direction_logits = []
    gen_word_logits = []
    dependents = [-1 for _ in word_ids]

    for action in actions:  
      transition_logit = None

      s0 = stack[-1] if len(stack) > 0 else 0

      pred_action = data_utils._ESH

      position = nn_utils.extract_feature_positions(buffer_index, s0,
          stack_next=self.stack_next)
      encoded_features = nn_utils.select_features(encoder_features, position, self.use_cuda)
      features = encoded_features
      
      head_ind = torch.LongTensor([1 if stack_has_parent and stack_has_parent[-1] else 0]).view(1, 1)

      if len(stack) > 0: # allowed to ra or la
        transition_logit = self.transition_model(features)

        if buffer_index == sent_length:
          assert action == data_utils._RE or action == data_utils._LA
          pred_action = data_utils._ERE
        else:
          if action == data_utils._SH:
            pred_action = data_utils._ESH
          elif action == data_utils._RA:
            pred_action = data_utils._ERA
          else:
            pred_action = data_utils._ERE

        if pred_action == data_utils._ERE:
          if stack_has_parent[-1]:
            assert action == data_utils._RE
          elif buffer_index == sent_length:
            assert action == data_utils._RE or action == data_utils._LA
          else:
            assert action == data_utils._LA

      transition_logits.append(transition_logit)

      # excecute action
      if action == data_utils._SH or action == data_utils._RA:
        if action == data_utils._RA:
          stack_has_parent.append(True)
        else:
          stack_has_parent.append(False)
        stack.append(buffer_index)
        buffer_index += 1
      else:  
        assert len(stack) > 0
        child = stack.pop()
        has_right_arc = stack_has_parent.pop()
        if has_right_arc or buffer_index == sent_length: # reduce
          dependents[child] = stack[-1]
        else: # left-arc
          dependents[child] = buffer_index

    return transition_logits, actions, dependents, dependents


  def _viterbi_algorithm(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)
    eps = np.exp(-10) # used to avoid division by 0

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length, 2])
    ra_log_probs = np.zeros([seq_length, seq_length, 2])
    re_log_probs = np.zeros([seq_length, seq_length, 2])
    word_log_probs = np.empty([sent_length, sent_length, 2])
    word_log_probs.fill(-np.inf)

    # batch feature computation
    enc_features = nn_utils.batch_feature_selection(encoder_features, seq_length,
        self.use_cuda, stack_next=self.stack_next)
    features = enc_features
    num_items = enc_features.size()[0]
    # TODO worry about expanded features etc later

    tr_log_probs_list = nn_utils.to_numpy(self.log_normalize(
        self.transition_model(features)).view(num_items, self.num_transitions))
    word_dist = self.log_normalize(self.word_model(features)).view(num_items, -1)

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        for c in range(2):
          shift_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ESH]
          ra_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERA]
          re_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERE]
          if j < sent_length:
            word_log_probs[i, j, c] = nn_utils.to_numpy(
                word_dist[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = len(word_ids)
    table = np.empty([table_size, 2, table_size, 2, table_size])
    table.fill(-np.inf) # log probabilities

    split_indexes = np.zeros((table_size, 2, table_size, 2, table_size), 
                             dtype=np.int)
    headedness = np.zeros((table_size, 2, table_size, 2, table_size), 
                          dtype=np.int)
    headedness.fill(0) # default
    
    # first word prob 
    if self.stack_next:
      table[0, 0, 0, 0, 1] = 0
    else:
      init_features = nn_utils.select_features(encoder_features, [0, 0], 
                                               self.use_cuda)
      init_word_dist = self.log_normalize(self.word_model(init_features).view(-1))
      table[0, 0, 0, 0, 1] = nn_utils.to_numpy(init_word_dist[word_ids[1]])
    
    # word probs
    for i in range(sent_length-1):
      for j in range(i+1, sent_length):
        for c in range(2):
          table[i, c, j, 0, j+1] = shift_log_probs[i, j, c] 
          table[i, c, j, 1, j+1] = ra_log_probs[i, j, c] 
          if self.word_model is not None:
            table[i, c, j, 0, j+1] += word_log_probs[i, j, c]
            table[i, c, j, 1, j+1] += word_log_probs[i, j, c]

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        for c in range(1 if i == 0 else 2):
          temp_right = []
          temp_headed = []
          for k in range(i+1, j):
            score0 = table[i, c, k, 0, j] + re_log_probs[k, j, 0]
            score1 = table[i, c, k, 1, j] + re_log_probs[k, j, 1]
            if score1 > score0 or (j == sent_length):
              temp_right.append(score1)
              temp_headed.append(1)
            else:
              temp_right.append(score0)
              temp_headed.append(0)

          for l in range(max(i, 1)):
            for b in range(1 if i == 0 else 2):
              block_scores = []
              for k in range(i+1, j):
                item_score = table[l, b, i, c, k] + temp_right[k - (i+1)]
                block_scores.append(item_score)
              ind = np.argmax(block_scores)
              k = ind + i + 1
              table[l, b, i, c, j] = block_scores[ind]
              split_indexes[l, b, i, c, j] = k
              headedness[l, b, i, c, j] = temp_headed[ind]

    def backtrack_path(l, b, i, c, j):
      """ Find action sequence for best path. """
      if i == j - 1:
        if c == 0:
          return [data_utils._SH]
        else:
          return [data_utils._RA]
      else:
        k = split_indexes[l, b, i, c, j]
        headed = headedness[l, b, i, c, j]
        if j == sent_length:
          assert headed
        act = data_utils._LA if headed == 0 else data_utils._RE
        return (backtrack_path(l, b, i, c, k) 
                + backtrack_path(i, c, k, headed, j) + [act])

    actions = backtrack_path(0, 0, 0, 0, sent_length)
    return self._decode_action_sequence(encoder_features, word_ids, actions)


  def neg_log_likelihood(self, sentence):
    batch_size = sentence.size()[1]
    encoder_state = self.encoder_model.init_hidden(batch_size)
    encoder_features = self.encoder_model(sentence, encoder_state)

    loss = -torch.sum(self._inside_algorithm_iterative(
        encoder_features, sentence, batch_size))
    return loss

  def forward(self, sentence):
    # for decoding
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    return self._viterbi_algorithm(encoder_features, word_ids)

