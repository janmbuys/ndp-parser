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

class ArcHybridSup(nn.Module):
  """Stack-based generative model with dynamic programming inference.""" 

  def __init__(self, vocab_size, num_relations, embedding_size, hidden_size, 
               num_layers, dropout, init_weight_range, bidirectional, non_lin,
               gen_non_lin, generative, stack_next, use_more_features, use_cuda):
    super(ArcHybridSup, self).__init__()
    self.use_cuda = use_cuda
    self.stack_next = stack_next
    num_features = 2
    num_dir_features = 3 if use_more_features else 2
    self.more_context = use_more_features
    self.generative = generative
    self.num_relations = num_relations
    self.bidirectional = bidirectional

    self.feature_size = (hidden_size*2 if bidirectional else hidden_size)

    self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, embedding_size, 
        hidden_size, num_layers, dropout, init_weight_range, 
        bidirectional=bidirectional, use_cuda=use_cuda)

    self.transition_model = binary_classifier.BinaryClassifier(num_features, 
        0, self.feature_size, hidden_size, non_lin, use_cuda) 
    self.direction_model = binary_classifier.BinaryClassifier(num_dir_features, 
        0, self.feature_size, hidden_size, non_lin, use_cuda) 

    if self.generative: 
      self.word_model = classifier.Classifier(num_features, 0, self.feature_size,
          hidden_size, vocab_size, gen_non_lin, use_cuda)
    else:
      self.word_model = None

    self.relation_model = classifier.Classifier(num_dir_features, 0, 
        self.feature_size, hidden_size, num_relations, non_lin, use_cuda)

    self.log_normalize = nn.LogSoftmax()
    self.binary_log_normalize = nn.LogSigmoid()

    self.criterion = nn.CrossEntropyLoss(size_average=False)
    self.binary_criterion = nn.BCEWithLogitsLoss(size_average=False)


  def inside_algorithm_cubic(self, encoder_features, sentence, batch_size):
    assert self.generative
    # enc feature dim length x batch x state_size
    # sentence dim length x batch
    sent_length = sentence.size()[0] - 1
    seq_length = sentence.size()[0]

    # dim [num_pairs, batch_size, 4, state_size]
    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    rev_features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, rev=True, stack_next=self.stack_next)
    num_pairs = features.size()[0]

    sh_log_probs_list = self.binary_log_normalize(-self.transition_model(features)).view(num_pairs, batch_size)
    re_rev_log_probs_list = self.binary_log_normalize(self.transition_model(rev_features)).view(num_pairs, batch_size)

    # dim [num_pairs*batch_size, output_size] -> break dim
    word_distr_list = self.log_normalize(self.word_model(features)).view(
        num_pairs, batch_size, -1) 

    # enumerate indexes
    inds_table = np.zeros((seq_length, seq_length), dtype=np.int)
    counter = 0
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        #inds_table[i, j-1 if self.stack_next else j] = counter
        inds_table[i, j] = counter
        counter += 1

    def get_feature_index(i, j):
      #return inds_table[i, j-1 if self.stack_next else j]
      return inds_table[i, j]

    # rather enumerate indexes for the reverse
    rev_inds_table = np.zeros((seq_length, seq_length), dtype=np.int)
    counter = 0
    for j in range(1, seq_length):
      for i in range(j):
        #rev_inds_table[i, j-1 if self.stack_next else j] = counter
        rev_inds_table[i, j] = counter
        counter += 1

    def get_rev_feature_index(i, j):
      #return rev_inds_table[i, j-1 if self.stack_next else j]
      return rev_inds_table[i, j]

    table_size = sentence.size()[0]
    table = nn_utils.to_var(torch.FloatTensor(table_size, table_size, 
        batch_size).fill_(-np.inf), self.use_cuda)

    word_prob_list = nn_utils.to_var(torch.FloatTensor(table_size, table_size, 
        batch_size).fill_(-np.inf), self.use_cuda)

    # word probs
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
            sentence[j if self.stack_next else j+1].view(-1, 1)).view(-1)
        word_prob_list[i, j] = sh_log_probs_list[index] + word_prob

    for j in range(1, sent_length):
      table[j, j+1] = 0

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        start_ind = get_rev_feature_index(i+1, j)
        end_ind = get_rev_feature_index(j-1, j) + 1
              
        # Super vectorization 
        sh_probs = word_prob_list[i, i+1:j]
        re_probs = re_rev_log_probs_list[start_ind:end_ind]
        shre_probs = sh_probs + re_probs
        all_block_scores = table[i, i+1:j] + table[i+1:j, j] + shre_probs
        table[i, j] = nn_utils.log_sum_exp(all_block_scores, 0)

        #temp_right = table[i, i+1:j, j] + re_probs #.view(1, -1, batch_size)
        #all_block_scores = (table[0:max(i, 1), i, i+1:j]
        #                    + temp_right.expand(max(i, 1), gap - 1, batch_size))
        #print(all_block_scores.size())
        #print(nn_utils.log_sum_exp(all_block_scores, 1).size())
        #table[0:max(i, 1), i, j] = nn_utils.log_sum_exp(all_block_scores, 1)
     
    # actually needed, but not trained
    if self.stack_next:
      final_score = re_rev_log_probs_list[get_rev_feature_index(0, sent_length)]
    else:
      final_score = 0
    return table[0, sent_length] + final_score


  def inside_algorithm(self, encoder_features, sentence, batch_size):
    assert self.generative
    # enc feature dim length x batch x state_size
    # sentence dim length x batch
    sent_length = sentence.size()[0] - 1
    seq_length = sentence.size()[0]

    # dim [num_pairs, batch_size, 4, state_size]
    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    rev_features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, rev=True, stack_next=self.stack_next)
    num_pairs = features.size()[0]

    sh_log_probs_list = self.binary_log_normalize(-self.transition_model(features)).view(num_pairs, batch_size)
    re_rev_log_probs_list = self.binary_log_normalize(self.transition_model(rev_features)).view(num_pairs, batch_size)

    # dim [num_pairs*batch_size, output_size] -> break dim
    word_distr_list = self.log_normalize(self.word_model(features)).view(
        num_pairs, batch_size, -1) 

    # enumerate indexes
    inds_table = np.zeros((seq_length, seq_length), dtype=np.int)
    counter = 0
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        #inds_table[i, j-1 if self.stack_next else j] = counter
        inds_table[i, j] = counter
        counter += 1

    def get_feature_index(i, j):
      #return inds_table[i, j-1 if self.stack_next else j]
      return inds_table[i, j]

    # rather enumerate indexes for the reverse
    rev_inds_table = np.zeros((seq_length, seq_length), dtype=np.int)
    counter = 0
    for j in range(1, seq_length):
      for i in range(j):
        #rev_inds_table[i, j-1 if self.stack_next else j] = counter
        rev_inds_table[i, j] = counter
        counter += 1

    def get_rev_feature_index(i, j):
      #return rev_inds_table[i, j-1 if self.stack_next else j]
      return rev_inds_table[i, j]

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
            sentence[j if self.stack_next else j+1].view(-1, 1)).view(-1)
        table[i, j, j+1] = sh_log_probs_list[index] + word_prob

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        start_ind = get_rev_feature_index(i+1, j)
        end_ind = get_rev_feature_index(j-1, j) + 1
              
        # Super vectorization 
        re_probs = re_rev_log_probs_list[start_ind:end_ind]
        temp_right = table[i, i+1:j, j] + re_probs #.view(1, -1, batch_size)
        all_block_scores = (table[0:max(i, 1), i, i+1:j]
                            + temp_right.expand(max(i, 1), gap - 1, batch_size))
        #print(all_block_scores.size())
        #print(nn_utils.log_sum_exp(all_block_scores, 1).size())
        table[0:max(i, 1), i, j] = nn_utils.log_sum_exp(all_block_scores, 1)
     
    # actually needed, but not trained
    if self.stack_next:
      final_score = re_rev_log_probs_list[get_rev_feature_index(0, sent_length)]
    else:
      final_score = 0
    return table[0, 0, sent_length] + final_score

  #TODO implement this, but might prioritize other things first
  def viterbi_generate(self, encoder_features, word_ids):
    assert self.generative
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    reduce_log_probs = np.zeros([seq_length, seq_length])

    if self.more_context:
      la_log_probs = np.zeros([seq_length, seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length, seq_length])
    else:
      la_log_probs = np.zeros([seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length])

    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    if self.more_context:
      dir_features = nn_utils.batch_feature_selection_more_context(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
      ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(dir_features)))
      la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(dir_features)))
    else:
      ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(features)))
      la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(features)))
    word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    dir_counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter]
        reduce_log_probs[i, j] = re_log_probs_list[counter]
        if self.more_context:
          for k in range(max(i, 1)):
            la_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + la_dir_log_probs_list[dir_counter])
            ra_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + ra_dir_log_probs_list[dir_counter])
            dir_counter += 1
        else:
          la_log_probs[i, j] = (re_log_probs_list[counter] 
                                + la_dir_log_probs_list[counter])
          ra_log_probs[i, j] = (re_log_probs_list[counter] 
                                + ra_dir_log_probs_list[counter])

        if j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = seq_length
    table = np.empty([table_size, table_size])
    table.fill(-np.inf) # log probabilities
    table_ned = np.empty([table_size, table_size]) # no end reduce
    table_ned.fill(-np.inf) # log probabilities

    split_indexes = np.zeros((table_size, table_size), dtype=np.int)
    split_indexes_ned = np.zeros((table_size, table_size), dtype=np.int)
    directions = np.zeros((table_size, table_size), dtype=np.int)
    directions.fill(data_utils._DRA) # default direction

    word_seq_score = 0

    for j in range(0, sent_length):
      if not self.stack_next and j == 0:
        init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
        init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
        table[0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])
        table_ned[0, 1] = table[0, 1]
        word_seq_score = table[0, 1]
      else:
        table[j, j+1] = 0
        table_ned[j, j+1] = 0

    ned_split_index = 0
    split_index_list = []

    for j in range(2, sent_length+1):
      for i in range(j-2, -1, -1):
        block_scores = []
        ned_block_scores = []
        block_directions = []
        for k in range(i+1, j):
          score = (table[i, k] + table[k, j] 
                   + shift_log_probs[i, k] + word_log_probs[i, k])
          if self.more_context:
            ra_prob = ra_log_probs[i, k, j]
            la_prob = la_log_probs[i, k, j]
          else:
            ra_prob = ra_log_probs[k, j]
            la_prob = la_log_probs[k, j]
          if ra_prob > la_prob or j == sent_length:
            score += ra_prob
            block_directions.append(data_utils._DRA)
          else:
            score += la_prob
            block_directions.append(data_utils._DLA)
          block_scores.append(score)
          ned_score = (table[0, k] + table[k, j]  #table_ned
                       + shift_log_probs[0, k] + word_log_probs[0, k])
          ned_block_scores.append(ned_score)

        ind = np.argmax(block_scores)
        table[i, j] = block_scores[ind]
        split_indexes[i, j] = ind + i + 1
        directions[i, j] = block_directions[ind]
        
        ned_ind = np.argmax(ned_block_scores)
        table_ned[i, j] = ned_block_scores[ned_ind]
        split_indexes_ned[i, j] = ned_ind + i + 1

      # want to generate j-1 (j for buffer_next)
      i = 0
      k = split_indexes_ned[i, j]
      while k < j-1:
        i = k
        k = split_indexes_ned[i, j]
      word_seq_score += word_log_probs[i, j-1]
      ned_split_index = i
      split_index_list.append(i)
      if j == sent_length and self.stack_next:
        for k in range(ned_split_index, -1, -1):
          word_seq_score += reduce_log_probs[k, j]

    #print(split_index_list)
    return word_seq_score


  def greedy_generate(self, encoder_features, word_ids, given_actions=None):
    assert self.generative
    #TODO later support partially given sentence
    stack = []
    buffer_index = 0
    sent_length = len(word_ids) - 1
    #print("sentence length %d" % sent_length) 
    self.eval()
    root_id = word_ids[0] #word_vocab.get_id('*root*')
    word_id = root_id

    num_actions = 0
    predicted_actions = []
    dependents = [-1]
    labels = [-1]
    sentence = []
    hidden = self.encoder_model.init_hidden(1)
    encoder_output = []

    greedy_word_loss = 0
    greedy_loss = 0
    has_eos = False

    if not self.stack_next:
      sentence.append(word_id)
      id_tensor = torch.LongTensor([word_id]).view(1, 1)
      embed = self.encoder_model.drop(self.encoder_model.embed(nn_utils.to_var(id_tensor,
              self.use_cuda, True)).view(1, 1, -1))
      output, hidden = self.encoder_model.rnn(embed, hidden)
      output = self.encoder_model.drop(output)
      encoder_output.append(output)
      dependents.append(-1)
      labels.append(-1)

    #while buffer_index < sent_length or len(stack) > 1:
    while not has_eos:
      #print(buffer_index)
      if given_actions is not None:
        assert len(given_actions) > num_actions
        action = given_actions[num_actions]
      else:
        action = data_utils._SH

      label = -1
      s0 = stack[-1] if len(stack) > 0 else 0

      if buffer_index > 0 or not self.stack_next:
        position = nn_utils.extract_feature_positions(
            buffer_index, s0, stack_next=self.stack_next)
        #nn_utils.select_features(encoder_output, position, self.use_cuda)
        features = torch.cat([encoder_output[ind] for ind in position], 0)

      def _test_logit(logit):
        return float(nn_utils.to_numpy(logit)) > 0

      if len(stack) > 1 or (self.stack_next and len(stack) > 0): # allowed to ra or la
        transition_logit = self.transition_model(features)
        if given_actions is not None:
          if buffer_index == sent_length:
            assert action == data_utils._RA
            sh_action = data_utils._SRE
          elif action == data_utils._SH:
            sh_action = data_utils._SSH
          else:
            sh_action = data_utils._SRE
        else:
          #sh_action = 1 if _test_logit(transition_logit) else 0
          tr_sample = torch.bernoulli(torch.sigmoid(transition_logit))
          sh_action = int(nn_utils.to_numpy(tr_sample)) 
        if self.generative:
          if sh_action == 1:
            transition_prob = self.binary_log_normalize(transition_logit)
          else:
            transition_prob = self.binary_log_normalize(-transition_logit)
          greedy_loss += nn_utils.to_numpy(transition_prob.view(1))

        if sh_action == data_utils._SRE:
          if given_actions is not None:
            if action == data_utils._LA:
              direc = data_utils._DLA  
            else:
              direc = data_utils._DRA
          else:
            # Note: don't predict direction
            #direction_logit = self.direction_model(features)  
            #direc = 1 if _test_logit(direction_logit) else 0
            #dir_sample = torch.bernoulli(torch.sigmoid(direction_logit))
            #direc = int(nn_utils.to_numpy(dir_sample)) 
            direc = data_utils._DRA

            if direc == data_utils._DLA:
              action = data_utils._LA
            else:
              action = data_utils._RA
        
        #if action != data_utils._SH:
        #  relation_logit = self.relation_model(features)      
        #  relation_logit_np = nn_utils.to_numpy(relation_logit)
        #  label = int(relation_logit_np.argmax(axis=1)[0])
        
      num_actions += 1
      if given_actions is None:
        predicted_actions.append(action)
      #print("action %d" % action)

      # excecute action
      if action == data_utils._SH:
        stack.append(buffer_index)
        # generate next word
        if buffer_index == 0 and self.stack_next:
          word_id = root_id
        else:
          logits = self.word_model(features)
          word_log_distr = self.log_normalize(logits).view(-1)
          word_distr = torch.nn.functional.softmax(logits.view(-1))
          word_sample = torch.multinomial(word_distr, 1).view(1)
          word_id = int(nn_utils.to_numpy(word_sample))

          greedy_word_loss += nn_utils.to_numpy(word_log_distr[word_id].view(1))
          greedy_loss += nn_utils.to_numpy(word_log_distr[word_id].view(1))
        if word_id == data_utils._EOS and not self.stack_next:
          has_eos = True
        else:
          sentence.append(word_id)
          # compute next hidden state
          id_tensor = torch.LongTensor([word_id]).view(1, 1)
          embed = self.encoder_model.drop(self.encoder_model.embed(nn_utils.to_var(id_tensor,
              self.use_cuda, True)).view(1, 1, -1))
          output, hidden = self.encoder_model.rnn(embed, hidden)
          output = self.encoder_model.drop(output)
          encoder_output.append(output)
           
          dependents.append(-1)
          labels.append(-1)
          buffer_index += 1
      else:  
        assert len(stack) > 0
        child = stack.pop()
        if len(stack) == 0:
            has_eos = True
        elif action == data_utils._LA:
          dependents[child] = buffer_index
        else:
          dependents[child] = stack[-1]
          #if len(stack) == 1 and len(sentence) > 1:
        #labels[child] = label

    actions = given_actions if given_actions is not None else predicted_actions
    return actions, dependents, labels, sentence, -greedy_loss[0]


  def greedy_decode(self, encoder_features, word_ids, given_actions=None):
    stack = []
    buffer_index = 0
    sent_length = len(word_ids) - 1
    #print("sentence length %d" % sent_length) 
    more_context = False # TODO temp self.more_context

    num_actions = 0
    predicted_actions = []
    dependents = [-1 for _ in word_ids]
    labels = [-1 for _ in word_ids]
    greedy_word_loss = 0
    greedy_loss = 0

    while (buffer_index < sent_length or len(stack) > 1 
           or (self.stack_next and len(stack) > 0)):
      #print(buffer_index)
      if buffer_index == sent_length and self.stack_next and len(stack) == 1:
        action = data_utils._LA
      elif given_actions is not None:
        assert num_actions < len(given_actions)
        action = given_actions[num_actions]
      else:
        action = data_utils._SH

      label = -1
      s0 = stack[-1] if len(stack) > 0 else 0
      s1 = stack[-2] if len(stack) > 1 else 0

      position = nn_utils.extract_feature_positions(
          buffer_index, s0, stack_next=self.stack_next)
      features = nn_utils.select_features(encoder_features[1], position, self.use_cuda)
      label_position = nn_utils.extract_feature_positions(
          buffer_index, s0, s1, more_context=more_context, stack_next=self.stack_next)
      label_features = nn_utils.select_features(encoder_features[1], label_position, self.use_cuda)

      def _test_logit(logit):
        return float(nn_utils.to_numpy(logit)) > 0

      if len(stack) > 1 or (self.stack_next and len(stack) > 0): # allowed to ra or la
        transition_logit = self.transition_model(features)
        if buffer_index == sent_length and len(stack) > 1:
          if given_actions is not None:
            assert action == data_utils._RA
          else:
            action = data_utils._RA
          sh_action = data_utils._SRE
        elif given_actions is not None:
          if action == data_utils._SH:
            sh_action = data_utils._SSH
          else:
            sh_action = data_utils._SRE
        else:
          #transition_logit = self.transition_model(features)
          sh_action = 1 if _test_logit(transition_logit) else 0
          
        if sh_action == 1:
          transition_prob = self.binary_log_normalize(transition_logit)
        else:
          transition_prob = self.binary_log_normalize(-transition_logit)
        greedy_loss += nn_utils.to_numpy(transition_prob.view(1))
        if self.stack_next and buffer_index == sent_length:
          greedy_word_loss += nn_utils.to_numpy(transition_prob.view(1))

        if sh_action == data_utils._SRE:
          if given_actions is not None:
            if action == data_utils._LA:
              direc = data_utils._DLA  
            else:
              direc = data_utils._DRA
          else:
            direction_logit = self.direction_model(label_features)  
            direc = 1 if _test_logit(direction_logit) else 0

            if direc == data_utils._DLA:
              action = data_utils._LA
            else:
              action = data_utils._RA
        
        if action != data_utils._SH:
          relation_logit = self.relation_model(label_features)      
          relation_logit_np = nn_utils.to_numpy(relation_logit)
          label = int(relation_logit_np.argmax(axis=1)[0])
        
      if self.generative and action == data_utils._SH:
        word_distr = self.log_normalize(self.word_model(features)).view(-1)
        if not self.stack_next or buffer_index > 0:
          word_id = word_ids[buffer_index if self.stack_next else buffer_index+1]
          greedy_word_loss += nn_utils.to_numpy(word_distr[word_id].view(1))
          greedy_loss += nn_utils.to_numpy(word_distr[word_id].view(1))

      num_actions += 1
      if given_actions is None:
        predicted_actions.append(action)
      #print("action %d" % action)

      # excecute action
      if action == data_utils._SH:
        stack.append(buffer_index)
        buffer_index += 1
      else:  
        assert len(stack) > 0
        child = stack.pop()
        if child > 0:
          if action == data_utils._LA or len(stack) == 0:
            dependents[child] = buffer_index
          else:
            dependents[child] = stack[-1]
        labels[child] = label
    if self.generative:
      loss = -greedy_loss[0]
    else:
      loss = 0

    actions = given_actions if given_actions is not None else predicted_actions
    return actions, dependents, labels, loss


  def backtrack_path(self, i, j, split_indexes, directions):
    """ Find action sequence for best path. """
    if i == j - 1:
      return [data_utils._SH]
    else:
      k = split_indexes[i, j]
      direct = directions[i, j]
      act = data_utils._LA if direct == data_utils._DLA else data_utils._RA
      return (self.backtrack_path(i, k, split_indexes, directions) +
              self.backtrack_path(k, j, split_indexes, directions) + [act])


  def incremental_inside_score(self, encoder_features, word_ids):
    assert self.generative
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    reduce_log_probs = np.zeros([seq_length, seq_length])

    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    dir_counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter]
        reduce_log_probs[i, j] = re_log_probs_list[counter]
        if j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = seq_length
    table = np.empty([table_size, table_size])
    table.fill(-np.inf) # log probabilities
    table_ned = np.empty([table_size, table_size]) # no end reduce
    table_ned.fill(-np.inf) # log probabilities

    word_seq_score = 0

    for j in range(0, sent_length):
      if not self.stack_next and j == 0:
        init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
        init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
        table[0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])
        table_ned[0, 1] = table[0, 1]
        word_seq_score = table[0, 1]
      else:
        table[j, j+1] = 0
        table_ned[j, j+1] = 0

    ned_split_index = 0
    split_index_list = []

    for j in range(2, sent_length+1):
      for i in range(j-2, -1, -1):
        block_score = -np.inf
        ned_block_score = -np.inf
        for k in range(i+1, j):
          score = (table[i, k] + table[k, j] 
                   + shift_log_probs[i, k] + word_log_probs[i, k]
                   + reduce_log_probs[k, j])
          block_score = np.logaddexp(block_score, score)
          ned_score = (table[0, k] + table[k, j]  # table_ned might be correct
                       + shift_log_probs[0, k] + word_log_probs[0, k])
          ned_block_score = np.logaddexp(ned_block_score, ned_score)

        table[i, j] = block_score
        table_ned[i, j] = ned_block_score

      # generating j-1 (j for buffer_next)
      word_seq_score = table_ned[0, j] #excluding final reduce

    return word_seq_score


  def incremental_viterbi_score(self, encoder_features, word_ids):
    assert self.generative
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    reduce_log_probs = np.zeros([seq_length, seq_length])

    if self.more_context:
      la_log_probs = np.zeros([seq_length, seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length, seq_length])
    else:
      la_log_probs = np.zeros([seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length])

    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    if self.more_context:
      dir_features = nn_utils.batch_feature_selection_more_context(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
      ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(dir_features)))
      la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(dir_features)))
    else:
      ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(features)))
      la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(features)))
    word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    dir_counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter]
        reduce_log_probs[i, j] = re_log_probs_list[counter]
        if self.more_context:
          for k in range(max(i, 1)):
            la_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + la_dir_log_probs_list[dir_counter])
            ra_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + ra_dir_log_probs_list[dir_counter])
            dir_counter += 1
        else:
          la_log_probs[i, j] = (re_log_probs_list[counter] 
                                + la_dir_log_probs_list[counter])
          ra_log_probs[i, j] = (re_log_probs_list[counter] 
                                + ra_dir_log_probs_list[counter])

        if j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = seq_length
    table = np.empty([table_size, table_size])
    table.fill(-np.inf) # log probabilities
    table_ned = np.empty([table_size, table_size]) # no end reduce
    table_ned.fill(-np.inf) # log probabilities

    split_indexes = np.zeros((table_size, table_size), dtype=np.int)
    split_indexes_ned = np.zeros((table_size, table_size), dtype=np.int)
    directions = np.zeros((table_size, table_size), dtype=np.int)
    directions.fill(data_utils._DRA) # default direction

    word_seq_score = 0

    for j in range(0, sent_length):
      if not self.stack_next and j == 0:
        init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
        init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
        table[0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])
        table_ned[0, 1] = table[0, 1]
        word_seq_score = table[0, 1]
      else:
        table[j, j+1] = 0
        table_ned[j, j+1] = 0

    ned_split_index = 0
    split_index_list = []

    for j in range(2, sent_length+1):
      for i in range(j-2, -1, -1):
        block_scores = []
        ned_block_scores = []
        block_directions = []
        for k in range(i+1, j):
          score = (table[i, k] + table[k, j] 
                   + shift_log_probs[i, k] + word_log_probs[i, k])
          if self.more_context:
            ra_prob = ra_log_probs[i, k, j]
            la_prob = la_log_probs[i, k, j]
          else:
            ra_prob = ra_log_probs[k, j]
            la_prob = la_log_probs[k, j]
          if ra_prob > la_prob or j == sent_length:
            score += ra_prob
            block_directions.append(data_utils._DRA)
          else:
            score += la_prob
            block_directions.append(data_utils._DLA)
          block_scores.append(score)
          ned_score = (table[0, k] + table[k, j]  #table_ned
                       + shift_log_probs[0, k] + word_log_probs[0, k])
          ned_block_scores.append(ned_score)

        ind = np.argmax(block_scores)
        table[i, j] = block_scores[ind]
        split_indexes[i, j] = ind + i + 1
        directions[i, j] = block_directions[ind]
        
        ned_ind = np.argmax(ned_block_scores)
        table_ned[i, j] = ned_block_scores[ned_ind]
        split_indexes_ned[i, j] = ned_ind + i + 1

      # want to generate j-1 (j for buffer_next)
      i = 0
      k = split_indexes_ned[i, j]
      while k < j-1:
        i = k
        k = split_indexes_ned[i, j]
      word_seq_score += word_log_probs[i, j-1]
      ned_split_index = i
      split_index_list.append(i)
      if j == sent_length and self.stack_next:
        for k in range(ned_split_index, -1, -1):
          word_seq_score += reduce_log_probs[k, j]

    #print(split_index_list)
    return word_seq_score


  # score outside computation graph
  def inside_score_cubic(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    reduce_log_probs = np.zeros([seq_length, seq_length])
    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter]
        reduce_log_probs[i, j] = re_log_probs_list[counter]

        if j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = seq_length
    table = np.empty([table_size, table_size])
    table.fill(-np.inf) # log probabilities

    for j in range(0, sent_length):
      if not self.stack_next and j == 0:
        init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
        init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
        table[0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])
      else:
        table[j, j+1] = 0

    for j in range(2, sent_length+1):
      for i in range(j-2, -1, -1):
        block_score = -np.inf
        block_scores = []
        for k in range(i+1, j):
          score = (table[i, k] + table[k, j] + reduce_log_probs[k, j] 
                   + shift_log_probs[i, k] + word_log_probs[i, k]) 
          block_score = np.logaddexp(block_score, score)
          block_scores.append(score)
        #print("sum %f" % float(block_score))
        #print("argmax %f" % float(block_scores[np.argmax(block_scores)]))
        #table[i, j] = block_scores[np.argmax(block_scores)]
        table[i, j] = block_score

    #print("sum % f" % float(table[0, sent_length]))
    #print("argmax % f" % float(table_max[0, sent_length]))
    if self.stack_next:
      final_score = reduce_log_probs[0, sent_length]
    else:
      final_score = 0
    return table[0, sent_length] + final_score


  # score outside computation graph
  def inside_score(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    reduce_log_probs = np.zeros([seq_length, seq_length])
    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter]
        reduce_log_probs[i, j] = re_log_probs_list[counter]

        if j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = seq_length
    table = np.empty([table_size, table_size, table_size])
    table.fill(-np.inf) # log probabilities

    # first word prob 
    if not self.stack_next:
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
      init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
      table[0, 0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])
    else:
      table[0, 0, 1] = 0

    for j in range(2, sent_length+1):
      for i in range(j-1):
        table[i, j-1, j] = shift_log_probs[i, j-1] + word_log_probs[i, j-1]
      for i in range(j-2, -1, -1):
        for l in range(max(i, 1)): # l=0 if i=0
          block_score = -np.inf
          block_scores = []
          for k in range(i+1, j):
            score = table[l, i, k] + table[i, k, j] + reduce_log_probs[k, j]
            block_score = np.logaddexp(block_score, score)
            block_scores.append(score)
          #table[l, i, j] = block_scores[np.argmax(block_scores)] 
          table[l, i, j] = block_score

    if self.stack_next:
      final_score = reduce_log_probs[0, sent_length]
    else:
      final_score = 0
    return table[0, 0, sent_length] + final_score


  def viterbi_decode_cubic(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)
    more_context = False # TODO temp self.more_context

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    if more_context:
      la_log_probs = np.zeros([seq_length, seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length, seq_length])
    else:
      la_log_probs = np.zeros([seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length])

    if self.generative:
      word_log_probs = np.empty([sent_length, sent_length])
      word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    if more_context:
      dir_features = nn_utils.batch_feature_selection_more_context(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
      ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(dir_features)))
      la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(dir_features)))
    else:
      ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(features)))
      la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(features)))

    if self.generative:
      word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    dir_counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter]
        if more_context:
          for k in range(max(i, 1)):
            la_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + la_dir_log_probs_list[dir_counter])
            ra_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + ra_dir_log_probs_list[dir_counter])
            dir_counter += 1
        else:
          la_log_probs[i, j] = (re_log_probs_list[counter] 
                                + la_dir_log_probs_list[counter])
          ra_log_probs[i, j] = (re_log_probs_list[counter] 
                                + ra_dir_log_probs_list[counter])

        if self.generative and j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = seq_length
    table = np.empty([table_size, table_size])
    table.fill(-np.inf) # log probabilities
    split_indexes = np.zeros((table_size, table_size), dtype=np.int)
    directions = np.zeros((table_size, table_size), dtype=np.int)
    directions.fill(data_utils._DRA) # default direction

    for j in range(0, sent_length):
      if self.generative and not self.stack_next and j == 0:
        # first word prob 
        init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
        init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
        table[0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])
      else:
        table[j, j+1] = 0

    for j in range(2, sent_length+1):
      for i in range(j-2, -1, -1):
        block_scores = [] # adjust indexes after collecting scores
        block_directions = []
        for k in range(i+1, j):
          score = table[i, k] + table[k, j] + shift_log_probs[i, k] 
          if self.generative:
            score += word_log_probs[i, k]
          if more_context:
            ra_prob = ra_log_probs[i, k, j]
            la_prob = la_log_probs[i, k, j]
          else:
            ra_prob = ra_log_probs[k, j]
            la_prob = la_log_probs[k, j]
          if ra_prob > la_prob or j == sent_length:
            score += ra_prob
            block_directions.append(data_utils._DRA)
          else:
            score += la_prob
            block_directions.append(data_utils._DLA)
          block_scores.append(score)
        ind = np.argmax(block_scores)
        k = ind + i + 1
        table[i, j] = block_scores[ind]
        split_indexes[i, j] = k
        directions[i, j] = block_directions[ind]

    def backtrack_path(i, j):
      """ Find action sequence for best path. """
      if i == j - 1:
        return [data_utils._SH]
      else:
        k = split_indexes[i, j]
        direct = directions[i, j]
        act = data_utils._LA if direct == data_utils._DLA else data_utils._RA
        return (backtrack_path(i, k) + backtrack_path(k, j) + [act])
  
    actions = backtrack_path(0, sent_length)
    return self.greedy_decode(encoder_features, word_ids, actions)


  def viterbi_decode(self, encoder_features, word_ids):
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    if self.more_context:
      la_log_probs = np.zeros([seq_length, seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length, seq_length])
    else:
      la_log_probs = np.zeros([seq_length, seq_length])
      ra_log_probs = np.zeros([seq_length, seq_length])

    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    dir_features = nn_utils.batch_feature_selection_more_context(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(dir_features)))
    la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(dir_features)))

    if self.generative:
      word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    dir_counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter] # counter, 0]
        if self.more_context:
          for k in range(max(i, 1)):
            la_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + la_dir_log_probs_list[dir_counter])
            ra_log_probs[k, i, j] = (re_log_probs_list[counter] 
                                  + ra_dir_log_probs_list[dir_counter])
            dir_counter += 1
        else:
          la_log_probs[i, j] = (re_log_probs_list[counter] 
                                + la_dir_log_probs_list[counter])
          ra_log_probs[i, j] = (re_log_probs_list[counter] 
                                + ra_dir_log_probs_list[counter])

        if self.generative and j < sent_length:
          word_log_probs[i, j] = nn_utils.to_numpy(
              word_distr_list[counter, word_ids[j if self.stack_next else j+1]])
        counter += 1

    table_size = seq_length
    table = np.empty([table_size, table_size, table_size])
    table.fill(-np.inf) # log probabilities
    split_indexes = np.zeros((table_size, table_size, table_size), dtype=np.int)
    directions = np.zeros((table_size, table_size, table_size), dtype=np.int)
    directions.fill(data_utils._DRA) # default direction

    # first word prob 
    if self.generative and not self.stack_next:
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
      init_word_distr = self.log_normalize(self.word_model(init_features).view(-1))
      table[0, 0, 1] = nn_utils.to_numpy(init_word_distr[word_ids[1]])
    else:
      table[0, 0, 1] = 0

    for j in range(2, sent_length+1):
      for i in range(j-1):
        score = shift_log_probs[i, j-1]
        if self.generative:
          score += word_log_probs[i, j-1]
        table[i, j-1, j] = score
      for i in range(j-2, -1, -1):
        for l in range(max(i, 1)): # l=0 if i=0
          block_scores = [] # adjust indexes after collecting scores
          block_directions = []
          for k in range(i+1, j):
            score = table[l, i, k] + table[i, k, j] 
            if self.more_context:
              ra_prob = ra_log_probs[i, k, j]
              la_prob = la_log_probs[i, k, j]
            else:
              ra_prob = ra_log_probs[k, j]
              la_prob = la_log_probs[k, j]
            if ra_prob > la_prob or j == sent_length:
              score += ra_prob
              block_directions.append(data_utils._DRA)
            else:
              score += la_prob
              block_directions.append(data_utils._DLA)
            block_scores.append(score)
          ind = np.argmax(block_scores)
          k = ind + i + 1
          table[l, i, j] = block_scores[ind]
          split_indexes[l, i, j] = k
          directions[l, i, j] = block_directions[ind]

    def backtrack_path(l, i, j):
      """ Find action sequence for best path. """
      if i == j - 1:
        return [data_utils._SH]
      else:
        k = split_indexes[l, i, j]
        direct = directions[l, i, j]
        act = data_utils._LA if direct == data_utils._DLA else data_utils._RA
        return (backtrack_path(l, i, k) + backtrack_path(i, k, j) + [act])
  
    actions = backtrack_path(0, 0, sent_length)
    return self.greedy_decode(encoder_features, word_ids, actions)
  

  def neg_log_likelihood_train(self, sentence):
    batch_size = sentence.size()[1]
    encoder_state = self.encoder_model.init_hidden(batch_size)
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    batch_loss = self.inside_algorithm(
        encoder_features, sentence, batch_size)
    #batch_loss = self.inside_algorithm_cubic(
    #    encoder_features, sentence, batch_size)
    batch_loss[batch_loss != batch_loss] = 0
    loss = -torch.sum(batch_loss)
    return loss


  def neg_log_likelihood(self, sentence):
    batch_size = sentence.size()[1]
    encoder_state = self.encoder_model.init_hidden(batch_size)
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    #For values (remove .data for training)
    loss = -torch.sum(self.inside_algorithm_cubic(
        encoder_features, sentence, batch_size)).data[0]

    #loss = -self.inside_score(encoder_features, word_ids)
    #loss = -self.inside_score_cubic(encoder_features, word_ids)
    return loss


  def viterbi_neg_log_likelihood(self, sentence):
    batch_size = sentence.size()[1]
    encoder_state = self.encoder_model.init_hidden(batch_size)
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    loss = -self.incremental_inside_score(encoder_features, word_ids)
    #loss = -self.incremental_viterbi_score(encoder_features, word_ids)
    return loss


  def lookup_features(self, encoder_features, feature_pos):
    feat_seq_size = int(feature_pos.size()[0])
    batch_size = int(feature_pos.size()[1])
    num_slots = int(feature_pos.size()[2])
    
    poss = []
    for i in range(num_slots):
      # assume pair features for now
      p_pos = feature_pos[:,:,i].unsqueeze(2).expand(feat_seq_size, 
          batch_size, self.feature_size)
      p_features = torch.gather(encoder_features[1], 0, p_pos)
      poss.append(p_features)

    #right_pos = feature_pos[:,:,1].unsqueeze(2).expand(feat_seq_size, 
    #    batch_size, self.feature_size)
    #right_features = torch.gather(encoder_features[1], 0, right_pos)
    #features = torch.stack((left_features, right_features), 2)

    features = torch.stack(poss, 2)
    return features # size [num_features, batch_size, 2, embed_size]


  # features dimension [num_features, batch_size, 2]
  # This should be the same for AH and AE - perhaps have a shared based class?
  # tuple order (transition_action, [direction], word_gen, relation_label)
  def joint_neg_log_likelihood(self, sentence, feature_inds, prediction_vars):
    # directions and relation_labels assumed to use same features

    batch_size = sentence.size()[1]
    sent_length = sentence.size()[0] - 1
    encoder_state = self.encoder_model.init_hidden(batch_size)
    encoder_features = self.encoder_model(sentence, encoder_state)
    
    slot_features = [self.lookup_features(encoder_features, feature_ind)
                     for feature_ind in feature_inds]

    # assume all models are used for now
    # return logits vector length [seq_len*batch_size, num_classes]
    transition_logit = self.transition_model(slot_features[0])
    direction_logit = self.direction_model(slot_features[2])
    relation_logits = self.relation_model(slot_features[2])
    if self.generative:
      gen_word_logits = self.word_model(slot_features[1])

    tr_loss = self.binary_criterion(transition_logit.view(-1),
        prediction_vars[0].view(-1))
    dir_loss = self.binary_criterion(direction_logit.view(-1),
        prediction_vars[1].view(-1))
    rel_loss = self.criterion(relation_logits, prediction_vars[3].view(-1))
    loss = tr_loss + dir_loss + rel_loss
    if self.generative:
      word_loss = self.criterion(gen_word_logits, prediction_vars[2].view(-1))
      loss += word_loss

    #loss = loss/(sent_length*batch_size) #TODO test - LM does not average
    return loss


  def generate(self, sentence, viterbi=True, gold_actions=None):
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    if gold_actions is not None:
      return self.greedy_generate(encoder_features, word_ids,
          given_actions=gold_actions)
    #elif viterbi:
    #  return self.viterbi_generate(encoder_features, word_ids)
    else:
      return self.greedy_generate(encoder_features, word_ids)


  def forward(self, sentence, viterbi=True, gold_actions=None):
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    if gold_actions is not None:
      return self.greedy_decode(encoder_features, word_ids,
          given_actions=gold_actions)
    elif viterbi:
      return self.viterbi_decode_cubic(encoder_features, word_ids)
      #return self.viterbi_decode(encoder_features, word_ids)
    else:
      return self.greedy_decode(encoder_features, word_ids)


  def decompose_transitions(self, actions):
    stackops = []
    directions = []
    
    for action in actions:
      if action == data_utils._SH:
        stackops.append(data_utils._SSH)
      else:
        stackops.append(data_utils._SRE)
        if action == data_utils._LA:
          directions.append(data_utils._DLA)
        else:  
          directions.append(data_utils._DRA)
    
    return stackops, directions


  def oracle(self, conll):
    """Training oracle for single parsed sentence, not in computation graph."""
    stack = data_utils.ParseForest([])
    buf = data_utils.ParseForest([conll[0]])
    buffer_index = 0
    sent_length = len(conll)

    num_children = [0 for _ in conll]
    for token in conll:
      if token.parent_id >= 0:
        num_children[token.parent_id] += 1

    actions = []
    labels = []
    words = []
    tr_features = []
    gen_features = []
    label_features = []

    while buffer_index < sent_length or len(stack) > 1:
      action = data_utils._SH
      s0 = stack.roots[-1].id if len(stack) > 0 else 0
      s1 = stack.roots[-2].id if len(stack) > 1 else 0

      if len(stack) > 1: # allowed to ra or la
        if buffer_index == sent_length:
          action = data_utils._RA
        else:
          if stack.roots[-1].parent_id == buffer_index:
            if len(stack.roots[-1].children) == num_children[s0]:
              action = data_utils._LA 
          elif stack.roots[-1].parent_id == s1:
            if len(stack.roots[-1].children) == num_children[s0]:
              action = data_utils._RA 
   
      if buffer_index > 0 or not self.stack_next:
        position = nn_utils.extract_feature_positions(buffer_index, s0,
          more_context=False, stack_next=self.stack_next) #(stack ind, buffer ind)
        tr_features.append(position)
        actions.append(action)
      
      if action == data_utils._SH:
        if buffer_index+1 < sent_length or self.stack_next:
          word = conll[buffer_index if self.stack_next else buffer_index+1].word_id
        else:
          word = data_utils._EOS
        if buffer_index > 0 or not self.stack_next:
          gen_features.append(position)
          words.append(word)
      else:
        label_position = nn_utils.extract_feature_positions(buffer_index, s0, s1,
          more_context=self.more_context, stack_next=self.stack_next) #([s1], s0, buffer ind)
        label = stack.roots[-1].relation_id
        label_features.append(label_position) # also used for direction
        labels.append(label)

      # excecute action
      if action == data_utils._SH:
        stack.roots.append(buf.roots[0]) 
        buffer_index += 1
        if buffer_index == sent_length:
          buf = data_utils.ParseForest([])
        else:
          buf = data_utils.ParseForest([conll[buffer_index]])
      else:  
        assert len(stack) > 0
        child = stack.roots.pop()
        if action == data_utils._LA:
          buf.roots[0].children.append(child) 
        else:
          stack.roots[-1].children.append(child)

    # final reduce for stack_next
    if self.stack_next:
      action = data_utils._LA
      s0 = stack.roots[-1].id if len(stack) > 0 else 0
      s1 = stack.roots[-2].id if len(stack) > 1 else 0
      position = nn_utils.extract_feature_positions(buffer_index, s0,
          more_context=False, stack_next=True)
      tr_features.append(position)
      actions.append(action)
      label_position = nn_utils.extract_feature_positions(buffer_index, s0, s1,
          more_context=self.more_context, stack_next=self.stack_next) #([s1], s0, buffer ind)
      label = stack.roots[-1].relation_id # not actually used
      label_features.append(label_position)
      labels.append(label)

    stackops, directions = self.decompose_transitions(actions)
    predictions = (list(map(lambda x: torch.FloatTensor(x).view(-1, 1), 
                            [stackops, directions]))
                   + list(map(lambda x: torch.LongTensor(x).view(-1, 1), 
                            [words, labels])))

    features = list(map(lambda x: torch.LongTensor(x).view(-1, 1, 2), 
                        [tr_features, gen_features]))
    features.append(torch.LongTensor(label_features).view(-1, 1, 
        3 if self.more_context else 2))
    return actions, predictions, features 

