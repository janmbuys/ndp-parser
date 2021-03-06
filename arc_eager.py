# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source

import numpy as np

import torch
import torch.nn as nn

import rnn_encoder
import classifier
import binary_classifier
import data_utils
import nn_utils
import transition_system as tr

class ArcEagerTransitionSystem(tr.TransitionSystem):
  def __init__(self, vocab_size, num_relations,
      embedding_size, hidden_size, num_layers, dropout, init_weight_range, 
      bidirectional, more_context, non_lin, gen_non_lin,
      predict_relations, generative, decompose_actions, embed_only, 
      embed_only_gen, stack_next, batch_size, use_cuda, model_path, 
      load_model, late_reduce_oracle):
    assert not more_context
    num_transitions = 3
    num_features = 2 # TODO not including headed feature
    super(ArcEagerTransitionSystem, self).__init__(vocab_size, num_relations,
        num_features, num_transitions, 0, 0, embedding_size, hidden_size, 
        num_layers, dropout, init_weight_range, bidirectional, non_lin,
        gen_non_lin, predict_relations, generative, decompose_actions, 
        embed_only, stack_next, batch_size, use_cuda, model_path, load_model)
    self.more_context = False
    self.generate_actions = [data_utils._SH, data_utils._RA]
    self.late_reduce_oracle = late_reduce_oracle
    self.embed_only_gen = embed_only_gen

    if False: # TODO remove
      if load_model:
        assert model_path != ''
        model_fn = model_path + '_headembed.pt'
        with open(model_fn, 'rb') as f:
          self.embed_headed = torch.load(f)
      else:  
        # embed binary features
        self.embed_headed = nn.Embedding(2, self.feature_size)
        self.embed_shift = nn.Embedding(2, self.feature_size) #TODO
        self.init_weights(init_weight_range)
       
      if use_cuda:
        self.embed_headed.cuda()
        if not load_model: #TODO
          self.embed_shift.cuda()

  def init_weights(self, initrange=0.1):
    self.embed_headed.weight.data.uniform_(-initrange, initrange)
    self.embed_shift.weight.data.uniform_(-initrange, initrange)

  def store_model(self, path):
    super(ArcEagerTransitionSystem, self).store_model(path)
    if False:
      model_fn = path + '_headembed.pt'
      with open(model_fn, 'wb') as f:
        torch.save(self.embed_headed, f)

  def map_transitions(self, actions):
    nactions = []
    for action in actions:
      if action == data_utils._SH:
        nactions.append(data_utils._ESH)
      elif action == data_utils._RA:
        nactions.append(data_utils._ERA)
      else:
        nactions.append(data_utils._ERE)
    return nactions 

  def decompose_transitions(self, actions):
    stackops = []
    arcops = []
    
    for action in actions:
      if action == data_utils._SH:
        stackops.append(data_utils._SSH)
        arcops.append(data_utils._LSH)
      elif action == data_utils._RA:
        stackops.append(data_utils._SSH)
        arcops.append(data_utils._LRA)
      elif action == data_utils._LA:
        stackops.append(data_utils._SRE)
        arcops.append(None) # data_utils._ULA)
      else:
        stackops.append(data_utils._SRE)
        arcops.append(None) # data_utils._URE)
   
    return stackops, arcops

  def greedy_decode(self, conll, encoder_features, given_actions=None):
    stack = data_utils.ParseForest([])
    stack_has_parent = []
    stack_parent_relation = []
    buf = data_utils.ParseForest([conll[0]])
    buffer_index = 0
    sent_length = len(conll)

    num_actions = 0
    predicted_actions = []
    labels = []
    words = []
    transition_logits = []
    direction_logits = []
    gen_word_logits = []
    relation_logits = []
    normalize = nn.Sigmoid()

    while buffer_index < sent_length or len(stack) > 1:
      transition_logit = None
      relation_logit = None
      direction_logit = None

      if given_actions is not None:
        assert len(given_actions) > num_actions
        action = given_actions[num_actions]
      else:
        action = data_utils._SH
      pred_action = data_utils._ESH

      label = -1
      s0 = stack.roots[-1].id if len(stack) > 0 else 0

      position = nn_utils.extract_feature_positions(buffer_index, s0, stack_next=self.stack_next)
      features = nn_utils.select_features(encoder_features[1], position, self.use_cuda)
      #head_ind = torch.LongTensor([1 if stack_has_parent and stack_has_parent[-1] else 0]).view(1, 1)
      # head_feat = self.embed_headed(nn_utils.to_var(head_ind, self.use_cuda)) 
      # features = torch.cat((enc_features, head_feat), 0) #TODO
      #features = enc_features

      if len(stack) > 0: # allowed to ra or la
        if buffer_index == sent_length:
          if given_actions is not None:
            assert action == data_utils._RE or action == data_utils._LA
          pred_action = data_utils._ERE
        else:
          transition_logit = self.transition_model(features)
          if given_actions is not None:
            if action == data_utils._SH:
              pred_action = data_utils._ESH
            elif action == data_utils._RA:
              pred_action = data_utils._ERA
            else:
              pred_action = data_utils._ERE
            if self.direction_model is not None:
              if pred_action != data_utils._ERE:
                direction_logit = self.direction_model(features)  
          elif self.direction_model is not None:
            #TODO instead of sigmoid can just test > 0 if not scoring
            transition_sigmoid = self.binary_normalize(transition_logit)
            transition_sigmoid_np = transition_sigmoid.type(torch.FloatTensor).data.numpy()
            sh_action = int(np.round(transition_sigmoid_np)[0])
            if sh_action == data_utils._SSH:
              direction_logit = self.direction_model(features)  
              direction_sigmoid = self.binary_normalize(direction_logit)
              direction_sigmoid_np = direction_sigmoid.type(torch.FloatTensor).data.numpy()
              direction = int(np.round(direction_sigmoid_np)[0])
              if direction == data_utils._LSH:
                pred_action = data_utils._ESH
              else:
                pred_action = data_utils._ERA
            else:
              pred_action = data_utils._ERE
          else:   
            transition_logit_np = transition_logit.type(torch.FloatTensor).data.numpy()
            pred_action = int(transition_logit_np.argmax(axis=1)[0])

        if given_actions is not None:
          if pred_action == data_utils._ERE:
            if stack_has_parent[-1]:
              assert action == data_utils._RE
            elif buffer_index == sent_length:
              assert action == data_utils._RE or action == data_utils._LA
            else:
              assert action == data_utils._LA
        else:      
          if pred_action == data_utils._ESH:
            action = data_utils._SH
          elif pred_action == data_utils._ERA:
            action = data_utils._RA
          elif stack_has_parent[-1] or buffer_index == sent_length:
            action = data_utils._RE
          else:
            action = data_utils._LA

        if self.relation_model is not None:
          # Need it for shift, so just as well do it for everything
          relation_logit = self.relation_model(features)      
          relation_logit_np = relation_logit.type(torch.FloatTensor).data.numpy()
          label = int(relation_logit_np.argmax(axis=1)[0])
        
      if self.word_model is not None and action == data_utils._SH:
        if self.embed_only_gen:
          gen_features = nn_utils.select_features(encoder_features[0], position, self.use_cuda)
          word_logit = self.word_model(gen_features)
        else:
          word_logit = self.word_model(features)
        gen_word_logits.append(word_logit)
        if buffer_index+1 < sent_length:
          word = conll[buffer_index+1].word_id
        else:
          word = data_utils._EOS
      else:
        word = -1
        if self.word_model is not None:
          gen_word_logits.append(None)

      if self.word_model is not None:
        words.append(word)

      transition_logits.append(transition_logit)
      direction_logits.append(transition_logit)
      num_actions += 1
      predicted_actions.append(pred_action) 
      if self.relation_model is not None:
        relation_logits.append(relation_logit)
        labels.append(label)

      # excecute action
      if action == data_utils._SH or action == data_utils._RA:
        child = buf.roots[0]
        if action == data_utils._RA:
          stack_has_parent.append(True)
        else:
          stack_has_parent.append(False)
        stack_parent_relation.append(label)

        stack.roots.append(child)
        buffer_index += 1
        if buffer_index == sent_length:
          buf = data_utils.ParseForest([])
        else:
          buf = data_utils.ParseForest([conll[buffer_index]])
      else:  
        assert len(stack) > 0
        child = stack.roots.pop()
        has_right_arc = stack_has_parent.pop()
        stack_label = stack_parent_relation.pop()
        if has_right_arc or buffer_index == sent_length: # reduce
          stack.roots[-1].children.append(child) 
          conll[child.id].pred_parent_id = stack.roots[-1].id
          if self.relation_model is not None:
            conll[child.id].pred_relation_ind = stack_label
        else: # left-arc
          buf.roots[0].children.append(child) 
          conll[child.id].pred_parent_id = buf.roots[0].id
          if self.relation_model is not None:
            conll[child.id].pred_relation_ind = label

    return conll, transition_logits, direction_logits, predicted_actions, relation_logits, labels, gen_word_logits, words


  def viterbi_decode(self, conll, encoder_features):
    sent_length = len(conll) # includes root, but not eos

    # compute all sh/re and word probabilities
    seq_length = sent_length + 1
    shift_log_probs = np.zeros([seq_length, seq_length, 2])
    ra_log_probs = np.zeros([seq_length, seq_length, 2])
    re_log_probs = np.zeros([seq_length, seq_length, 2])
    word_log_probs = np.empty([sent_length, sent_length, 2])
    word_log_probs.fill(-np.inf)

    # batch feature computation
    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    num_items = features.size()[0]

    # dim [num_items, 2 (headedness), batch_size, num_features, feature_size]
    #encoded_features = enc_features.view(num_items, 1, 1, 2, self.feature_size).expand(
    #  num_items, 2, 1, 2, self.feature_size)
    #features = enc_features

    # expand to add headedness feature
    #head_ind0 = torch.LongTensor(num_items).fill_(0)
    #head_ind1 = torch.LongTensor(num_items).fill_(1)
            
    #head_feat0 = self.embed_headed(nn_utils.to_var(head_ind0, self.use_cuda)) 
    #head_feat1 = self.embed_headed(nn_utils.to_var(head_ind1, self.use_cuda)) 
    #head_features = torch.cat((head_feat0.view(num_items, 1, 1, 1, -1), 
    #                           head_feat1.view(num_items, 1, 1, 1, -1)), 1)
    #features = torch.cat((encoded_features, head_features), 3) #TODO

    #TODO this is breaking
    #features = encoded_features
    #tr_log_probs_list = nn_utils.to_numpy(self.log_normalize(
    #    self.transition_model(features)).view(num_items, 2, self.num_transitions))

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

    if self.word_model is not None:
      #word_dist = self.log_normalize(self.word_model(features)).view(
      #    num_items, 2, self.vocab_size)
      if self.embed_only_gen:
        gen_features = nn_utils.batch_feature_selection(encoder_features[0], 
            seq_length, self.use_cuda, stack_next=self.stack_next)
        word_dist = self.log_normalize(self.word_model(gen_features)).view(
            num_items, self.vocab_size)
      else:
        word_dist = self.log_normalize(self.word_model(features)).view(
            num_items, self.vocab_size)

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        for c in range(2):
          if self.decompose_actions:
            shift_log_probs[i, j] = (sh_log_probs_list[counter] 
                                     + sh_dir_log_probs_list[counter])
            ra_log_probs[i, j] = (sh_log_probs_list[counter] 
                                  + ra_dir_log_probs_list[counter])
            re_log_probs[i, j] = re_log_probs_list[counter] 
          else:
            shift_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ESH]
            ra_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERA]
            re_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERE]
          if self.word_model is not None and j < sent_length:
            if j < sent_length - 1:
              word_id = conll[j+1].word_id 
            else:
              word_id = data_utils._EOS
            word_log_probs[i, j, c] = nn_utils.to_numpy(word_dist[counter, word_id])
        counter += 1

    table_size = sent_length + 1
    table = np.empty([table_size, 2, table_size, 2, table_size])
    table.fill(-np.inf) # log probabilities

    split_indexes = np.zeros((table_size, 2, table_size, 2, table_size), 
                             dtype=np.int)
    headedness = np.zeros((table_size, 2, table_size, 2, table_size), 
                          dtype=np.int)
    headedness.fill(0) # default
    
    # first word prob 
    if self.word_model is not None:
      if self.embed_only_gen:
        init_features = nn_utils.select_features(encoder_features[0], [0, 0], self.use_cuda)
      else:
        init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
      init_word_dist = self.log_normalize(self.word_model(init_features).view(-1))
      table[0, 0, 0, 0, 1] = nn_utils.to_numpy(init_word_dist[conll[1].word_id])
    else:
      table[0, 0, 0, 0, 1] = 0
    
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
               # to remove spurious ambiguity: 
               # if act=RE j-k > 1, ensure headed=true 
               # (ie, not preceeded by LA)

    actions = backtrack_path(0, 0, 0, 0, sent_length)
    #print(table[0, 0, 0, 0, sent_length]) # scores should match
    return self.greedy_decode(conll, encoder_features, actions)


  def inside_score(self, conll, encoder_features):
    assert self.word_model is not None
    sent_length = len(conll) # includes root, but not eos

    # compute all sh/re and word probabilities
    seq_length = sent_length + 1
    shift_log_probs = np.zeros([seq_length, seq_length, 2])
    ra_log_probs = np.zeros([seq_length, seq_length, 2])
    re_log_probs = np.zeros([seq_length, seq_length, 2])
    word_log_probs = np.empty([sent_length, sent_length, 2])
    word_log_probs.fill(-np.inf)

    # batch feature computation
    features = nn_utils.batch_feature_selection(encoder_features[1], seq_length,
        self.use_cuda, stack_next=self.stack_next)
    num_items = features.size()[0]

    # dim [num_items, 2 (headedness), batch_size, num_features, feature_size]
    #encoded_features = enc_features.view(num_items, 1, 1, 2, self.feature_size).expand(
    #    num_items, 2, 1, 2, self.feature_size)

    # expand to add headedness feature
    #head_ind0 = torch.LongTensor(num_items).fill_(0)
    #head_ind1 = torch.LongTensor(num_items).fill_(1)
            
    #head_feat0 = self.embed_headed(nn_utils.to_var(head_ind0, self.use_cuda)) 
    #head_feat1 = self.embed_headed(nn_utils.to_var(head_ind1, self.use_cuda)) 
    #head_features = torch.cat((head_feat0.view(num_items, 1, 1, 1, -1), 
    #                           head_feat1.view(num_items, 1, 1, 1, -1)), 1)
    #features = torch.cat((encoded_features, head_features), 3) #TODO
    #features = encoded_features
    #features = enc_features

    #tr_log_probs_list = nn_utils.to_numpy(self.log_normalize(
    #    self.transition_model(features)).view(num_items, 2, self.num_transitions))
    #word_dist = self.log_normalize(self.word_model(features)).view(
    #        num_items, 2, self.vocab_size)

    if self.decompose_actions:
      transition_logit = self.transition_model(features)
      direction_logit = self.direction_model(features)

      sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(
        -transition_logit).view(num_items))
      sh_ra_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(
        direction_logit).view(num_items)) + sh_log_probs_list
      sh_sh_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(
        -direction_logit).view(num_items)) + sh_log_probs_list
      re_log_probs_list = nn_utils.to_numpy(self.binary_log_normalize(
        transition_logit).view(num_items))
    else:  
      tr_log_probs_list = nn_utils.to_numpy(self.log_normalize(
          self.transition_model(features)).view(num_items, self.num_transitions))

    if self.embed_only_gen:
      gen_features = nn_utils.batch_feature_selection(encoder_features[0], 
          seq_length, self.use_cuda, stack_next=self.stack_next)
      word_dist = self.log_normalize(self.word_model(gen_features)).view(
          num_items, self.vocab_size)
    else:
      word_dist = self.log_normalize(self.word_model(features)).view(
          num_items, self.vocab_size)

    counter = 0 
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        for c in range(2):
          if self.decompose_actions:
            shift_log_probs[i, j, c] = sh_sh_log_probs_list[counter]
            ra_log_probs[i, j, c] = sh_ra_log_probs_list[counter]
            re_log_probs[i, j, c] = re_log_probs_list[counter]
          else:
            shift_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ESH]
            ra_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERA]
            re_log_probs[i, j, c] = tr_log_probs_list[counter, data_utils._ERE]
          if j < sent_length:
            if j < sent_length - 1:
              word_id = conll[j+1].word_id 
            else:
              word_id = data_utils._EOS
            word_log_probs[i, j, c] = nn_utils.to_numpy(word_dist[counter, word_id])
        counter += 1

    table_size = sent_length + 1
    table = np.empty([table_size, 2, table_size, 2, table_size])
    table.fill(-np.inf) # log probabilities

    if self.embed_only_gen:
      init_features = nn_utils.select_features(encoder_features[0], [0, 0], self.use_cuda)
    else:
      init_features = nn_utils.select_features(encoder_features[1], [0, 0], self.use_cuda)
    init_word_dist = self.log_normalize(self.word_model(init_features).view(-1))
    
    # word probs
    table[0, 0, 0, 0, 1] = nn_utils.to_numpy(init_word_dist[conll[1].word_id])
    for i in range(sent_length-1):
      for j in range(i+1, sent_length):
        for c in range(2):
          table[i, c, j, 0, j+1] = (shift_log_probs[i, j, c] 
                                    + word_log_probs[i, j, c])
          table[i, c, j, 1, j+1] = (ra_log_probs[i, j, c] 
                                    + word_log_probs[i, j, c])

    for gap in range(2, sent_length+1):
      for i in range(sent_length+1-gap):
        j = i + gap
        for c in range(1 if i == 0 else 2):
          temp_right = []
          for k in range(i+1, j):
            score0 = table[i, c, k, 0, j] + re_log_probs[k, j, 0]
            score1 = table[i, c, k, 1, j] + re_log_probs[k, j, 1]
            if j == sent_length:
              temp_right.append(score1)
            else:
              temp_right.append(np.logaddexp(score0, score1))

          for l in range(max(i, 1)):
            for b in range(1 if i == 0 else 2):
              block_score = -np.inf
              for k in range(i+1, j):
                item_score = table[l, b, i, c, k] + temp_right[k - (i+1)]
                block_score = np.logaddexp(block_score, item_score)

              table[l, b, i, c, j] = block_score
    return table[0, 0, 0, 0, sent_length] 


  def oracle(self, conll, encoder_features):
    # Training oracle for single parsed sentence.
    stack = data_utils.ParseForest([])
    stack_has_parent = []
    buf = data_utils.ParseForest([conll[0]])
    buffer_index = 0
    sent_length = len(conll)

    count_left_parent = [False for _ in conll]
    count_left_children = [0 for _ in conll]

    has_left_parent = [False for _ in conll]
    num_left_children = [0 for _ in conll]
    num_children = [0 for _ in conll]

    for token in conll:
      num_children[token.parent_id] += 1
      if token.id < token.parent_id:
        num_left_children[token.parent_id] += 1
      else:
        has_left_parent[token.id] = True

    actions = []
    labels = []
    words = []
    features = []
    gen_features = []

    while buffer_index < sent_length or len(stack) > 1:
      feature = None
      action = data_utils._SH
      s0 = stack.roots[-1].id if len(stack) > 0 else 0
      
      if len(stack) > 0: # allowed to ra or la
        if buffer_index == sent_length:
          action = data_utils._RE
        elif stack.roots[-1].parent_id == buffer_index:
          action = data_utils._LA 
        elif buf.roots[0].parent_id == stack.roots[-1].id:
          action = data_utils._RA 
        elif stack_has_parent[-1]:
          if (self.late_reduce_oracle and 
              ((has_left_parent[buffer_index] and not
                count_left_parent[buffer_index]) 
               or (count_left_children[buffer_index] <
                    num_left_children[buffer_index]))):
            action = data_utils._RE             
            assert len(stack.roots[-1].children) == num_children[s0]
          elif len(stack.roots[-1].children) == num_children[s0]:
            action = data_utils._RE 

      position = nn_utils.extract_feature_positions(buffer_index, s0,
          stack_next=self.stack_next) 
      feature = nn_utils.select_features(encoder_features[1], position, self.use_cuda)
     
      #head_ind = torch.LongTensor([1 if stack_has_parent and stack_has_parent[-1] else 0]).view(1, 1)
      #head_feat = self.embed_headed(nn_utils.to_var(head_ind, self.use_cuda)) 
      #feature = torch.cat((encoded_feature, head_feat), 0) #TODO
      #feature = encoded_feature
      #TODO also add shift binary feature

      features.append(feature)
      if self.embed_only_gen:
        gen_feature = nn_utils.select_features(encoder_features[0], position,
            self.use_cuda)
        gen_features.append(gen_feature)

      if action == data_utils._LA:
        label = stack.roots[-1].relation_id
      elif action == data_utils._RA:
        label = buf.roots[0].relation_id
      else:
        label = -1 
      
      if action == data_utils._SH or action == data_utils._RA:
        if buffer_index+1 < sent_length:
          word = conll[buffer_index+1].word_id
        else:
          word = data_utils._EOS
      else:
        word = -1
          
      actions.append(action)
      labels.append(label)
      words.append(word)

      # excecute action
      if action == data_utils._SH or action == data_utils._RA:
        child = buf.roots[0]
        if action == data_utils._RA:
          # this duplication might cause a problem
          #stack.roots[-1].children.append(child) 
          count_left_parent[buffer_index] = True
          stack_has_parent.append(True)
        else:
          stack_has_parent.append(False)
        stack.roots.append(child)
        buffer_index += 1
        if buffer_index == sent_length:
          buf = data_utils.ParseForest([])
        else:
          buf = data_utils.ParseForest([conll[buffer_index]])
      else:  
        assert len(stack) > 0
        child = stack.roots.pop()
        has_right_arc = stack_has_parent.pop()
        if action == data_utils._LA:
          buf.roots[0].children.append(child) 
          count_left_children[buffer_index] += 1
        else:
          assert has_right_arc
          stack.roots[-1].children.append(child) 
    return actions, words, labels, features, gen_features

