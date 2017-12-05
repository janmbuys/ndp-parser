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
               dropout, init_weight_range, non_lin, gen_non_lin,
               decompose_actions, stack_next, embed_only, embed_only_gen, 
               with_valency, use_cuda):
    super(ArcEagerDP, self).__init__()
    self.use_cuda = use_cuda
    self.stack_next = stack_next
    self.generate_actions = [data_utils._SH, data_utils._RA]
    self.embed_only_gen = embed_only_gen
    self.decompose_actions = decompose_actions
    self.num_transitions = 3
    self.with_valency = with_valency
    num_features = 2

    if embed_only:
      self.encoder_model = embed_encoder.EmbedEncoder(vocab_size, embedding_size, 
          hidden_size, dropout, init_weight_range, use_cuda=use_cuda)
    else:
      self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, embedding_size, 
          hidden_size, num_layers, dropout, init_weight_range, bidirectional=False, use_cuda=use_cuda)

    feature_size = hidden_size
    if decompose_actions:
      self.transition_model = binary_classifier.BinaryClassifier(num_features,
          1 if with_valency else 0, feature_size, hidden_size, non_lin, 
          use_cuda) # sh/re
      self.direction_model = binary_classifier.BinaryClassifier(num_features,
          1 if with_valency else 0, feature_size, hidden_size, non_lin, 
          use_cuda) # sh/ra 
    else:
      self.transition_model = classifier.Classifier(num_features, 
          1 if with_valency else 0, feature_size, hidden_size, 
          self.num_transitions, non_lin, use_cuda) 

    self.word_model = classifier.Classifier(num_features, 
        2 if with_valency else 0, feature_size, hidden_size, vocab_size, 
        gen_non_lin, use_cuda)

    self.log_normalize = nn.LogSoftmax()
    self.binary_log_normalize = nn.LogSigmoid()


  def _inside_algorithm(self, encoder_features, sentence, batch_size):
    sent_length = sentence.size()[0] - 1
    seq_length = sentence.size()[0]

    # batch feature computation
    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    rev_features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, rev=True, stack_next=self.stack_next)
    num_items = features.size()[0]
    if self.embed_only_gen:
      gen_features = nn_utils.batch_feature_selection(encoder_features[0], 
          seq_length, self.use_cuda, stack_next=self.stack_next)
    else:
      gen_features = features

    if self.with_valency:
      if self.decompose_actions:
        sh_ra_log_probs_list = []
        sh_sh_log_probs_list = []
        re_rev_log_probs_list = []

        for c in range(2): 
          transition_logit = self.transition_model(features, c)
          direction_logit = self.direction_model(features, c)
          sh_log_probs_list = self.binary_log_normalize(-transition_logit).view(
              num_items, batch_size)
          sh_ra_log_probs_list.append(self.binary_log_normalize(direction_logit).view(num_items,
            batch_size) + sh_log_probs_list)
          sh_sh_log_probs_list.append(self.binary_log_normalize(-direction_logit).view(num_items,
            batch_size) + sh_log_probs_list)

          re_rev_log_probs_list.append(self.binary_log_normalize(self.transition_model(
              rev_features, c)).view(num_items, batch_size))
      else:  
        tr_log_probs_list = []
        re_rev_log_probs_list = []
        for c in range(2):
          tr_log_probs_list.append(self.log_normalize(self.transition_model(
              features, c)).view(num_items, batch_size, self.num_transitions))
          re_rev_log_probs_list.append(self.log_normalize(
            self.transition_model(rev_features, c)).view(num_items, 
              batch_size, self.num_transitions)[:,:,data_utils._ERE])

      word_distr_list = []
      for ind in range(4):
        word_distr_list.append(self.log_normalize(self.word_model(
            gen_features, ind)).view(num_items, batch_size, -1))

    else:
      if self.decompose_actions:
        transition_logit = self.transition_model(features)
        direction_logit = self.direction_model(features)
        sh_log_probs_list = self.binary_log_normalize(-transition_logit).view(num_items, batch_size)
        sh_ra_log_probs_list = self.binary_log_normalize(direction_logit).view(num_items, batch_size) + sh_log_probs_list
        sh_sh_log_probs_list = self.binary_log_normalize(-direction_logit).view(num_items, batch_size) + sh_log_probs_list

        re_rev_log_probs_list = self.binary_log_normalize(self.transition_model(
            rev_features)).view(num_items, batch_size)
      else:  
        # dim [num_items, batch_size, output_size]
        tr_log_probs_list = self.log_normalize(self.transition_model(features)).view(
            num_items, batch_size, self.num_transitions)
        re_rev_log_probs_list = self.log_normalize(self.transition_model(
            rev_features)).view(num_items, batch_size, self.num_transitions)[:,:,data_utils._ERE]

      word_distr_list = self.log_normalize(self.word_model(gen_features)).view(
          num_items, batch_size, -1)

    # enumerate indexes
    inds_table = np.zeros((seq_length, seq_length), dtype=np.int) #, 2
    counter = 0
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        #for c in range(2):
        inds_table[i, j-1 if self.stack_next else j] = counter #, c
        counter += 1 # c's use same features for now

    def get_feature_index(i, j, c=None): # independent of c
      return inds_table[i, j-1 if self.stack_next else j]

    # enumerate reverse indexes
    rev_inds_table = np.zeros((seq_length, seq_length), dtype=np.int) #, 2
    counter = 0
    for j in range(1, seq_length):
      for i in range(j):
        #for c in range(2):
        rev_inds_table[i, j-1 if self.stack_next else j] = counter
        counter += 1

    def get_rev_feature_index(i, j, c=None):
      return rev_inds_table[i, j-1 if self.stack_next else j]

    table_size = sentence.size()[0]
    table = nn_utils.to_var(torch.FloatTensor(table_size, 2, table_size, 2,
        table_size, batch_size).fill_(-np.inf), self.use_cuda)
   
    # word probs
    if self.stack_next:
      table[0, 0, 0, 0, 1] = nn_utils.to_var(torch.FloatTensor(batch_size).fill_(0), self.use_cuda)
    else: #TODO
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
      if self.with_valency:
        init_word_logit = self.word_model(init_features, 0)
      else:
        init_word_logit = self.word_model(init_features)
      init_word_distr = self.log_normalize(init_word_logit)
      table[0, 0, 0, 0, 1] = torch.gather(init_word_distr, 1,
                                          sentence[1].view(-1, 1))

    # could potentially parallize over i, but may not be worth it
    for i in range(sent_length-1): 
      for j in range(i+1, sent_length):
        index = get_feature_index(i, j)
        if self.with_valency:
          for c in range(2):
            word_prob0 = torch.gather(word_distr_list[2*c][index], 1,
                sentence[j if self.stack_next else j+1].view(-1, 1))
            word_prob1 = torch.gather(word_distr_list[2*c+1][index], 1,
                sentence[j if self.stack_next else j+1].view(-1, 1))

            if self.decompose_actions:
              table[i, c, j, 0, j+1] = (sh_sh_log_probs_list[c][index] 
                                        + word_prob0)
              table[i, c, j, 1, j+1] = (sh_ra_log_probs_list[c][index] 
                                        + word_prob1)
            else:
              table[i, c, j, 0, j+1] = (tr_log_probs_list[c][index, :, data_utils._ESH]
                  + word_prob0)
              table[i, c, j, 1, j+1] = (tr_log_probs_list[c][index, :, data_utils._ERA]
                  + word_prob1)

        else:
          word_prob = torch.gather(word_distr_list[index], 1,
              sentence[j if self.stack_next else j+1].view(-1, 1))

          for c in range(2):
            if self.decompose_actions:
              table[i, c, j, 0, j+1] = sh_sh_log_probs_list[index] + word_prob
              table[i, c, j, 1, j+1] = sh_ra_log_probs_list[index] + word_prob
            else:
              table[i, c, j, 0, j+1] = (tr_log_probs_list[index, :, data_utils._ESH]
                  + word_prob)
              table[i, c, j, 1, j+1] = (tr_log_probs_list[index, :, data_utils._ERA]
                  + word_prob)

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        h = 1 if i == 0 else 2 # number of indicators to be filled

        start_ind = get_rev_feature_index(i+1, j)
        end_ind = get_rev_feature_index(j-1, j) + 1
         
        scores = nn_utils.to_var(torch.FloatTensor(1, 1, h, gap-1, 2, batch_size), self.cuda)

        if self.with_valency:
          for c in range(h):
            scores[0, 0, c, :, 0] = (table[i, c, i+1:j, 0, j] 
                + re_rev_log_probs_list[0][start_ind:end_ind])
            scores[0, 0, c, :, 1] = (table[i, c, i+1:j, 1, j]
                + re_rev_log_probs_list[1][start_ind:end_ind])
        else:
          re_probs = re_rev_log_probs_list[start_ind:end_ind]
          for c in range(h): # need loop because we can't expand re_probs
            scores[0, 0, c, :, 0] = table[i, c, i+1:j, 0, j] + re_probs
            scores[0, 0, c, :, 1] = table[i, c, i+1:j, 1, j] + re_probs

        # dim [1, 1, h, gap-1, batch_size]
        if j == sent_length:
          temp_right = scores[:, :, :, :, 1]
        else:
          temp_right = nn_utils.log_sum_exp(scores, 4).squeeze(4)

        block_scores = (table[0:max(i,1), 0:h, i, 0:h, i+1:j] +
                        temp_right.expand(max(i,1), h, h, gap-1, batch_size))

        table[0:max(i,1), 0:h, i, 0:h, j] = nn_utils.log_sum_exp(block_scores, 3)
    return table[0, 0, 0, 0, sent_length] #TODO calculate final reduce prob


  def _decode_action_sequence(self, encoder_features, word_ids, actions):
    stack = []
    stack_has_parent = []
    buffer_index = 0
    sent_length = len(word_ids) - 1

    dependents = [-1 for _ in word_ids]
    greedy_word_loss = 0

    for action in actions:  
      pred_action = data_utils._ESH

      s0 = stack[-1] if len(stack) > 0 else 0
      position = nn_utils.extract_feature_positions(
          buffer_index, s0, stack_next=self.stack_next)
      features = nn_utils.select_features(encoder_features[1], position, self.use_cuda)

      if len(stack) > 0: # allowed to ra or la
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

      if action == data_utils._SH or action == data_utils._RA:
        word_distr = self.log_normalize(self.word_model(features)).view(-1)
        word_id = word_ids[buffer_index if self.stack_next else buffer_index+1]
        greedy_word_loss += nn_utils.to_numpy(word_distr[word_id])

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

    return actions, dependents, greedy_word_loss


  def _viterbi_algorithm(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length, 2])
    ra_log_probs = np.zeros([seq_length, seq_length, 2])
    re_log_probs = np.zeros([seq_length, seq_length, 2])
    word_log_probs = np.empty([sent_length, sent_length, 4])
    word_log_probs.fill(-np.inf)

    # batch feature computation
    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    num_items = features.size()[0]

    if self.embed_only_gen:
       gen_features = nn_utils.batch_feature_selection(encoder_features[0], 
           seq_length, self.use_cuda, stack_next=self.stack_next)
    else:
      gen_features = features

    if self.with_valency:
      if self.decompose_actions:
        re_log_probs_list = []
        sh_log_probs_list = []
        ra_dir_log_probs_list = []
        sh_dir_log_probs_list = []

        for c in range(2): 
          transition_logit = self.transition_model(features, c)
          direction_logit = self.direction_model(features, c)
          re_log_probs_list.append(nn_utils.to_numpy(self.binary_log_normalize(transition_logit)))
          sh_log_probs_list.append(nn_utils.to_numpy(self.binary_log_normalize(-transition_logit)))
          ra_dir_log_probs_list.append(nn_utils.to_numpy(self.binary_log_normalize(direction_logit)))
          sh_dir_log_probs_list.append(nn_utils.to_numpy(self.binary_log_normalize(-direction_logit)))
      else:
        tr_log_probs_list = []
        for c in range(2):
          tr_log_probs_list.append(nn_utils.to_numpy(self.log_normalize(
              self.transition_model(features, c)).view(num_items,
                  self.num_transitions)))
      
      word_distr_list = []
      for ind in range(4):
        word_distr_list.append(self.log_normalize(self.word_model(
            gen_features, ind)).view(num_items, -1))
    else:
      if self.decompose_actions:
        transition_logit = self.transition_model(features)
        direction_logit = self.direction_model(features)
        re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(transition_logit))
        sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-transition_logit))
        ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(direction_logit))
        sh_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-direction_logit))
      else:
        tr_log_probs_list = nn_utils.to_numpy(self.log_normalize(
            self.transition_model(features)).view(num_items, self.num_transitions))

      word_distr_list = self.log_normalize(self.word_model(gen_features)).view(num_items, -1)

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        for c in range(2):
          if self.with_valency:
            if self.decompose_actions:
              shift_log_probs[i, j, c] = (sh_log_probs_list[c][counter] 
                                       + sh_dir_log_probs_list[c][counter])
              ra_log_probs[i, j, c] = (sh_log_probs_list[c][counter] 
                                    + ra_dir_log_probs_list[c][counter])
              re_log_probs[i, j, c] = re_log_probs_list[c][counter] 
            else:
              shift_log_probs[i, j, c] = tr_log_probs_list[c][counter, data_utils._ESH]
              ra_log_probs[i, j, c] = tr_log_probs_list[c][counter, data_utils._ERA]
              re_log_probs[i, j, c] = tr_log_probs_list[c][counter, data_utils._ERE]
          else:
            if self.decompose_actions:
              shift_log_probs[i, j, c] = (sh_log_probs_list[counter] 
                                       + sh_dir_log_probs_list[counter])
              ra_log_probs[i, j, c] = (sh_log_probs_list[counter] 
                                    + ra_dir_log_probs_list[counter])
              re_log_probs[i, j, c] = re_log_probs_list[counter] 
            else:
              shift_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ESH]
              ra_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERA]
              re_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERE]
            
        for ind in range(4):
          if j < sent_length:
            if self.with_valency:
              word_log_probs[i, j, ind] = nn_utils.to_numpy(
                  word_distr_list[ind][counter, word_ids[j if self.stack_next else j+1]])
            else:
              word_log_probs[i, j, ind] = nn_utils.to_numpy(
                  word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
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
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], 
                                               self.use_cuda)
      if self.with_valency:
        init_word_logit = self.word_model(init_features, 0)
      else:
        init_word_logit = self.word_model(init_features)
      init_word_dist = self.log_normalize(init_word_logit.view(-1))
      table[0, 0, 0, 0, 1] = nn_utils.to_numpy(init_word_dist[word_ids[1]])
    
    # word probs
    for i in range(sent_length-1):
      for j in range(i+1, sent_length):
        for c in range(2):
          table[i, c, j, 0, j+1] = (shift_log_probs[i, j, c] 
                                    + word_log_probs[i, j, 2*c])
          table[i, c, j, 1, j+1] = (ra_log_probs[i, j, c] 
                                    + word_log_probs[i, j, 2*c+1])

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

    loss = -torch.sum(self._inside_algorithm(encoder_features, sentence, 
                                             batch_size))
    return loss

  def forward(self, sentence):
    # for decoding
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    return self._viterbi_algorithm(encoder_features, word_ids)

