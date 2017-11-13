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
               gen_non_lin, generative, stack_next, use_cuda):
    super(ArcHybridSup, self).__init__()
    self.use_cuda = use_cuda
    self.stack_next = stack_next
    num_features = 2
    self.generative = generative
    self.num_relations = num_relations
    self.bidirectional = bidirectional

    self.feature_size = (hidden_size*2 if bidirectional else hidden_size)

    self.encoder_model = rnn_encoder.RNNEncoder(vocab_size, embedding_size, 
        hidden_size, num_layers, dropout, init_weight_range, 
        bidirectional=bidirectional, use_cuda=use_cuda)

    self.transition_model = binary_classifier.BinaryClassifier(num_features, 
        0, self.feature_size, hidden_size, non_lin, use_cuda) 
    self.direction_model = binary_classifier.BinaryClassifier(num_features, 
        0, self.feature_size, hidden_size, non_lin, use_cuda) 

    if self.generative: 
      self.word_model = classifier.Classifier(num_features, 0, self.feature_size,
          hidden_size, vocab_size, gen_non_lin, use_cuda)
    else:
      self.word_model = None

    self.relation_model = classifier.Classifier(num_features, 0, 
        self.feature_size, hidden_size, num_relations, non_lin, use_cuda)

    self.log_normalize = nn.LogSoftmax()
    self.binary_log_normalize = nn.LogSigmoid()

    self.criterion = nn.CrossEntropyLoss(size_average=False)
    self.binary_criterion = nn.BCEWithLogitsLoss(size_average=False)

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
     
    final_score = re_rev_log_probs_list[get_rev_feature_index(0, sent_length)]
    return table[0, 0, sent_length] + final_score


  def greedy_decode(self, encoder_features, word_ids, given_actions=None):
    stack = []
    buffer_index = 0
    sent_length = len(word_ids) - 1
    #print("sentence length %d" % sent_length) 

    num_actions = 0
    predicted_actions = []
    dependents = [-1 for _ in word_ids]
    labels = [-1 for _ in word_ids]

    while buffer_index < sent_length or len(stack) > 1:
      #print(buffer_index)
      if given_actions is not None:
        assert len(given_actions) > num_actions
        action = given_actions[num_actions]
      else:
        action = data_utils._SH

      label = -1
      s0 = stack[-1] if len(stack) > 0 else 0

      position = nn_utils.extract_feature_positions(
          buffer_index, s0, stack_next=self.stack_next)
      features = nn_utils.select_features(encoder_features[1], position, self.use_cuda)

      def _test_logit(logit):
        return float(nn_utils.to_numpy(logit)) > 0

      if len(stack) > 1: # allowed to ra or la
        if buffer_index == sent_length:
          if given_actions is not None:
            assert action == data_utils._RA
          else:
            action = data_utils._RA
            #print("end ra")
        elif given_actions is not None:
          if action == data_utils._SH:
            sh_action = data_utils._SSH
          else:
            sh_action = data_utils._SRE
        else:
          transition_logit = self.transition_model(features)
          sh_action = 1 if _test_logit(transition_logit) else 0

        if sh_action == data_utils._SRE:
          if given_actions is not None:
            if action == data_utils._LA:
              direc = data_utils._DLA  
            else:
              direc = data_utils._DRA
          else:
            direction_logit = self.direction_model(features)  
            direc = 1 if _test_logit(direction_logit) else 0

            if direc == data_utils._DLA:
              action = data_utils._LA
            else:
              action = data_utils._RA
        
        if action != data_utils._SH:
          relation_logit = self.relation_model(features)      
          relation_logit_np = nn_utils.to_numpy(relation_logit)
          label = int(relation_logit_np.argmax(axis=1)[0])
        
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
        if action == data_utils._LA:
          dependents[child] = buffer_index
        else:
          dependents[child] = stack[-1]
        labels[child] = label

    actions = given_actions if given_actions is not None else predicted_actions

    return actions, dependents, labels
  

  def viterbi_decode(self, encoder_features, word_ids):
    #TODO extend to la/ra (direction model)
    sent_length = len(word_ids) - 1
    seq_length = len(word_ids)

    # compute all sh/re and word probabilities
    shift_log_probs = np.zeros([seq_length, seq_length])
    la_log_probs = np.zeros([seq_length, seq_length])
    ra_log_probs = np.zeros([seq_length, seq_length])

    word_log_probs = np.empty([sent_length, sent_length])
    word_log_probs.fill(-np.inf)

    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.transition_model(features)))
    sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.transition_model(features)))

    ra_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(self.direction_model(features)))
    la_dir_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(-self.direction_model(features)))

    if self.generative:
      word_distr_list = self.log_normalize(self.word_model(features))

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        shift_log_probs[i, j] = sh_log_probs_list[counter] # counter, 0]
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
  

  def neg_log_likelihood(self, sentence):
    batch_size = sentence.size()[1]
    encoder_state = self.encoder_model.init_hidden(batch_size)
    encoder_features = self.encoder_model(sentence, encoder_state)

    loss = -torch.sum(self.inside_algorithm(
        encoder_features, sentence, batch_size))
    return loss


  def lookup_features(self, encoder_features, feature_pos):
    feat_seq_size = int(feature_pos.size()[0])
    batch_size = int(feature_pos.size()[1])
    # assume pair features for now
    left_pos = feature_pos[:,:,0].unsqueeze(2).expand(feat_seq_size, 
        batch_size, self.feature_size) #TODO test without expand
    #print(encoder_features[1].size())
    #print(left_pos.size())
    left_features = torch.gather(encoder_features[1], 0, left_pos)
    right_pos = feature_pos[:,:,1].unsqueeze(2).expand(feat_seq_size, 
        batch_size, self.feature_size)
    right_features = torch.gather(encoder_features[1], 0, right_pos)
    features = torch.stack((left_features, right_features), 2)
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


  def forward(self, sentence, viterbi=True):
    encoder_state = self.encoder_model.init_hidden(1) # batch_size==1
    encoder_features = self.encoder_model(sentence, encoder_state)
    word_ids = [int(x) for x in sentence.view(-1).data]

    if viterbi:
      return self.viterbi_decode(encoder_features, word_ids)
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


  def oracle(self, conll, stack_next=False):
    """Training oracle for single parsed sentence, not in computation graph."""
    stack = data_utils.ParseForest([])
    buf = data_utils.ParseForest([conll[0]])
    buffer_index = 0
    sent_length = len(conll)

    num_children = [0 for _ in conll]
    for token in conll:
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
      s2 = stack.roots[-3].id if len(stack) > 2 else 0

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
   
      position = nn_utils.extract_feature_positions(buffer_index, s0,
          more_context=False, stack_next=stack_next) #(stack ind, buffer ind)
      tr_features.append(position)
      actions.append(action)
      
      if action == data_utils._SH:
        if buffer_index+1 < sent_length or self.stack_next:
          word = conll[buffer_index if self.stack_next else buffer_index+1].word_id
        else:
          word = data_utils._EOS
        gen_features.append(position)
        words.append(word)
      else:
        label = stack.roots[-1].relation_id
        label_features.append(position)
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

    stackops, directions = self.decompose_transitions(actions)
    predictions = (list(map(lambda x: torch.FloatTensor(x).view(-1, 1), 
                            [stackops, directions]))
                   + list(map(lambda x: torch.LongTensor(x).view(-1, 1), 
                            [words, labels])))

    features = list(map(lambda x: torch.LongTensor(x).view(-1, 1, 2), 
                        [tr_features, gen_features, label_features]))
    return predictions, features 
