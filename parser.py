# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source

import argparse
import math
import os
import random
import sys
import time

from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import rnn_encoder
import classifier
import binary_classifier
import data_utils
import nn_utils

def to_numpy(dist):
  return dist.type(torch.FloatTensor).data.numpy()

# Training oracle for single example
def train_oracle(conll, encoder_features, more_context=False):
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
  gen_features = []
  features = []

  while buffer_index < sent_length or len(stack) > 1:
    feature = None
    action = data_utils._SH
    b = 0
    s0 = stack.roots[-1].id if len(stack) > 0 else 0

    if len(stack) > 1: # allowed to ra or la
      s0 = stack.roots[-1].id 
      s1 = stack.roots[-2].id 
      if buffer_index == sent_length:
        action = data_utils._RA
        b = sent_length - 1 # last encoded word
      else:
        b = buf.roots[0].id # == buffer_index
        if stack.roots[-1].parent_id == b: # candidate la
          if len(stack.roots[-1].children) == num_children[s0]:
            action = data_utils._LA 
        elif stack.roots[-1].parent_id == s1: # candidate ra
          if len(stack.roots[-1].children) == num_children[s0]:
            action = data_utils._RA 

      if more_context:
        s2 = stack.roots[-3].id if len(stack) >= 3 else 0
        position = [s0, s1, s2, b]
      else:
        position = [s0, s1, b]

      if args.cuda:
        feature_positions = Variable(torch.LongTensor(position)).cuda()
      else:
        feature_positions = Variable(torch.LongTensor(position))
      feature = torch.index_select(encoder_features, 0, feature_positions)

    if args.cuda:
      gen_feature_positions = Variable(torch.LongTensor([s0, b])).cuda()
    else:
      gen_feature_positions = Variable(torch.LongTensor([s0, b]))
    gen_feature = torch.index_select(encoder_features, 0, gen_feature_positions)

    actions.append(action)
    features.append(feature)
    gen_features.append(gen_feature)
     
    label = -1 if action == data_utils._SH else stack.roots[-1].relation_id
    if action == data_utils._SH:
      if buffer_index+1 < sent_length:
        word = conll[buffer_index+1].word_id
      else:
        word = data_utils._EOS
    else:
      word = -1
        
    labels.append(label)
    words.append(word)

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
  return actions, words, labels, features, gen_features

#TODO use for all instances with this contruction
def select_features(input_features, indexes, use_cuda=False):
  if use_cuda:
    positions = Variable(torch.LongTensor(indexes)).cuda()
  else:
    positions = Variable(torch.LongTensor(indexes))
  return torch.index_select(input_features, 0, positions)


#TODO make generative optional
def viterbi_decode(conll, encoder_features, transition_model,
        word_model=None, direction_model=None, relation_model=None):
  sent_length = len(conll) # includes root, but not EOS
  log_normalize = nn.LogSoftmax()
  binary_normalize = nn.Sigmoid()

  # compute all sh/re and word probabilities
  reduce_probs = np.zeros([sent_length, sent_length])
  word_log_probs = np.empty([sent_length, sent_length])
  word_log_probs.fill(-np.inf)

  # loop over features, range does not include EOS
  for i in range(sent_length-1):
    for j in range(i+1, sent_length):
      gen_features = select_features(encoder_features, [i, j], args.cuda)      
      re_prob = to_numpy(binary_normalize(transition_model(gen_features)))[0]
      reduce_probs[i, j] = re_prob
      word_dist = log_normalize(word_model(gen_features).view(-1))
      if j < sent_length - 1:
        word_log_probs[i, j] = to_numpy(word_dist[conll[j+1].word_id])
      else:
        word_log_probs[i, j] = to_numpy(word_dist[data_utils._EOS])

  table = np.empty([sent_length+1, sent_length+1])
  table.fill(-np.inf) # log probabilities
  split_indexes = np.zeros([sent_length+1, sent_length+1])
  directions = np.empty([sent_length+1, sent_length+1])
  directions.fill(-1)

  # first word prob 
  init_features = select_features(encoder_features, [0, 0], args.cuda)
  word_dist = log_normalize(word_model(init_features).view(-1))
  table[0, 1] = to_numpy(word_dist[conll[1].word_id])

  #TODO make code cleaner later
  for j in range(2, sent_length+1):
    word_block_scores = []
    for i in range(j-1):
      score = (table[i, j-1] + np.log(1 - reduce_probs[i, j-1]) 
               + word_log_probs[i, j-1])  
      word_block_scores.append(score)
    k = np.argmax(word_block_scores)
    table[j-1, j] = word_block_scores[k]
    split_indexes[j-1, j] = k
    for i in range(j-2, -1, -1):
      block_scores = [] # adjust indexes after collecting scores
      block_directions = []
      for k in range(i+1, j):
        if j < sent_length:
          score = table[i, k] + table[k, j] + np.log(reduce_probs[k, j])
          if direction_model is not None:
            dir_features = select_features(encoder_features, [i, k, j], args.cuda)
            ra_prob = to_numpy(binary_normalize(direction_model(dir_features)))[0]
            if ra_prob < 0.5:
              score += np.log(1-ra_prob)
              block_directions.append(data_utils._DLA)
            else:
              score += np.log(ra_prob)
              block_directions.append(data_utils._DRA)
        else:    
          # has to right-arc
          score = table[i, k] + table[k, j] 
          if direction_model is not None:
            block_directions.append(data_utils._DRA)
        block_scores.append(score)
      ind = np.argmax(block_scores)
      k = ind + i + 1
      table[i, j] = block_scores[ind]
      split_indexes[i, j] = k
      if direction_model is not None:
        directions[i, j] = block_directions[ind]
  # TODO trace back path

  #TODO return conll, transition_logits, direction_logits, actions, relation_logits, labels

def greedy_decode(conll, encoder_features, transition_model,
        relation_model=None, direction_model=None, more_context=False):
  stack = data_utils.ParseForest([])
  buf = data_utils.ParseForest([conll[0]])
  buffer_index = 0
  sent_length = len(conll)

  actions = []
  labels = []
  transition_logits = []
  direction_logits = []
  relation_logits = []
  normalize = nn.Sigmoid()

  while buffer_index < sent_length or len(stack) > 1:
    transition_logit = None
    relation_logit = None
    direction_logit = None
    action = data_utils._SH
    label = -1

    if len(stack) > 1: # allowed to ra or la
      s0 = stack.roots[-1].id 
      s1 = stack.roots[-2].id 
      if buffer_index == sent_length:
        b = sent_length - 1
      else:
        b = buf.roots[0].id 
      
      if more_context:
        s2 = stack.roots[-3].id if len(stack) >= 3 else 0
        position = [s0, s1, s2, b]
      else:
        position = [s0, s1, b]

      if args.cuda:
        feature_positions = Variable(torch.LongTensor(position)).cuda()
      else:
        feature_positions = Variable(torch.LongTensor(position))
      features = torch.index_select(encoder_features, 0, feature_positions)

      if args.cuda:
        gen_feature_positions = Variable(torch.LongTensor([s0, b])).cuda()
      else:
        gen_feature_positions = Variable(torch.LongTensor([s0, b]))
      gen_features = torch.index_select(encoder_features, 0, gen_feature_positions)

      #TODO rather score transition and relation jointly for greedy choice

      if direction_model is not None: # find action from decomposed 
        transition_logit = transition_model(gen_features)
        if buffer_index == sent_length:
          sh_action = data_utils._SRE 
        else: #TODO check, but instead of sigmoid can just test > 0
          transition_sigmoid = normalize(transition_logit)
          transition_sigmoid_np = transition_sigmoid.type(torch.FloatTensor).data.numpy()
          sh_action = int(np.round(transition_sigmoid_np)[0])
        if sh_action == data_utils._SRE:
          direction_logit = direction_model(features)  
          direction_sigmoid = normalize(direction_logit)
          direction_sigmoid_np = direction_sigmoid.type(torch.FloatTensor).data.numpy()
          if buffer_index == sent_length:
            direc = data_utils._DRA
          else:
            direc = int(np.round(direction_sigmoid_np)[0])
          if direc == data_utils._DLA:
            action = data_utils._LA
          else:
            action = data_utils._RA
          direction_logits.append(direction_logit)
        else:
          action = data_utils._SH
          direction_logits.append(None)
      else:
        transition_logit = transition_model(features)
        transition_logit_np = transition_logit.type(torch.FloatTensor).data.numpy()
        if buffer_index == sent_length:
          action = data_utils._RA
        else:
          action = int(transition_logit_np.argmax(axis=1)[0])
      
      if relation_model is not None and action != data_utils._SH:
        relation_logit = relation_model(features)      
        relation_logit_np = relation_logit.type(torch.FloatTensor).data.numpy()
        label = int(relation_logit_np.argmax(axis=1)[0])
      
    actions.append(action)
    transition_logits.append(transition_logit)
    if relation_model is not None:
      relation_logits.append(relation_logit)
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
        conll[child.id].pred_parent_id = buf.roots[0].id
      else:
        stack.roots[-1].children.append(child)
        conll[child.id].pred_parent_id = stack.roots[-1].id
      if relation_model is not None:
        conll[child.id].pred_relation_ind = label
  #print(labels)
  return conll, transition_logits, direction_logits, actions, relation_logits, labels
 

def filter_logits(logits, targets, float_var=False):
  logits_filtered = []
  targets_filtered = []
  for logit, target in zip(logits, targets):
    # this should handle the case where not direction logits 
    # should be predicted
    if logit is not None and target is not None: 
      logits_filtered.append(logit)
      targets_filtered.append(target)

  #logits_filtered = [logit for logit in logits if logit is not None]
  #targets_filtered = [target for (logit, target) in zip(logits, targets)
  #                        if logit is not None]

  if logits_filtered:
    if args.cuda:
      #print(logits_filtered)
      #print(targets_filtered)
      if float_var:
        target_var = Variable(torch.FloatTensor(targets_filtered)).cuda()
      else:  
        target_var = Variable(torch.LongTensor(targets_filtered)).cuda()
    else:
      if float_var:
        target_var = Variable(torch.FloatTensor(targets_filtered))
      else:
        target_var = Variable(torch.LongTensor(targets_filtered))

    output = torch.cat(logits_filtered, 0)
    return output, target_var
  else:
    return None, None

def decompose_transitions(actions):
  stackops = []
  directions = []
  
  for action in actions:
    if action == data_utils._SH:
      stackops.append(data_utils._SSH)
      directions.append(None)
    else:
      stackops.append(data_utils._SRE)
      if action == data_utils._LA:
        directions.append(data_utils._DLA)
      else:  
        directions.append(data_utils._DRA)
  
  return stackops, directions


def get_sentence_batch(source, use_cuda, evaluation=False):
  if use_cuda:
    data = Variable(source.word_tensor[:-1], volatile=evaluation).cuda()
  else:
    data = Variable(source.word_tensor[:-1], volatile=evaluation)
  return data


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True,
                      help='Directory of Annotated CONLL files')
  parser.add_argument('--data_working_dir', required=True,
                      help='Working directory for data files')
  parser.add_argument('--working_dir', required=True,
                      help='Working directory for model output')
  parser.add_argument('--train_name',
                      help='Train file name (excluding .conll)',
                      default='train')
  parser.add_argument('--dev_name',
                      help='Dev file name (excluding .conll)', 
                      default='dev')
  parser.add_argument('--test_name',
                      help='Test file name (excluding .conll)', 
                      default='test')
  parser.add_argument('--test_file',
                      help='Raw text file to parse with trained model', 
                      metavar='FILE')
  parser.add_argument('--replicate_rnng_data', action='store_true', 
                      default=False)
  parser.add_argument('--reset_vocab', action='store_true', 
                      default=False)

  parser.add_argument('--embedding_size', type=int, default=128,
                      help='size of word embeddings')
  parser.add_argument('--hidden_size', type=int, default=128, 
                      help='humber of hidden units per layer')
  parser.add_argument('--num_layers', type=int, default=1, # original 2
                      help='number of layers')
  parser.add_argument('--lr', type=float, default=0.001, # default 20 check
                      help='initial learning rate')
  parser.add_argument('--grad_clip', type=float, default=5, # default 0.5 check
                      help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=100,
                      help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                      help='batch size') # original default 20
  parser.add_argument('--dropout', type=float, default=0.0, 
                      help='dropout rate')
  parser.add_argument('--bidirectional', action='store_true',
                      help='use bidirectional encoder')
  parser.add_argument('--generative', action='store_true',
                      help='use generative parser')
  parser.add_argument('--decompose_actions', action='store_true',
                      help='decompose action prediction to fit dynamic program')

  parser.add_argument('--relation_prediction', dest='predict_relations',
                      action='store_true', help='predict relation labels.')
  parser.add_argument('--no_relation_prediction', dest='predict_relations',
                      action='store_false', 
                      help='do not predict relation labels.')
  parser.set_defaults(predict_relations=True)

  parser.add_argument('--cuda', action='store_true',
                      help='use CUDA')
  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  parser.add_argument('--small_data', action='store_true',
                      help='use small version of dataset')
  parser.add_argument('--logging_interval', type=int, default=5000, 
                      metavar='N', help='report ppl every x steps')
  parser.add_argument('--save_model', type=str,  default='model.pt',
                      help='path to save the final model')

  args = parser.parse_args()
  assert not (args.generative and args.bidirectional), 'Bidirectional encoder invalid for generative model.'

  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    assert torch.cuda.is_available(), 'Cuda not available.'
    torch.cuda.manual_seed(args.seed)

  data_path = args.data_dir + '/' 
  data_working_path = args.data_working_dir + '/'
  working_path = args.working_dir + '/'

  # Prepare training data

  vocab_path = Path(data_working_path + 'vocab')
  if vocab_path.is_file() and not args.reset_vocab:
    sentences, word_vocab, pos_vocab, rel_vocab = data_utils.read_sentences_given_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data)
  else:     
    print('Preparing vocab')
    sentences, word_vocab, pos_vocab, rel_vocab = data_utils.read_sentences_create_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data)

  # Read dev and test files with given vocab
  dev_sentences, _, _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.dev_name, data_working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data)

  test_sentences, _,  _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.test_name, data_working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data)

  if args.small_data:
    sentences = sentences[:100]
    dev_sentences = dev_sentences[:100]

  #data_utils.create_length_histogram(sentences)

  # Extract static oracle sequences
  #for sent in sentences:
  #  actions, labels = data_utils.oracle(sent.conll)

  def train():
    vocab_size = len(word_vocab)
    num_relations = len(rel_vocab)
    num_transitions = 3
    num_features = 3 if args.generative or args.decompose_actions else 4
    num_gen_features = 2

    batch_size = args.batch_size

    # Build the model
    encoder_model = rnn_encoder.RNNEncoder(vocab_size, 
        args.embedding_size, args.hidden_size, args.num_layers, args.dropout,
        args.bidirectional, args.cuda)

    #TODO factorize features to work with dynamic program
    feature_size = (args.hidden_size*2 if args.bidirectional 
                    else args.hidden_size)
    #TODO add option for decomposition depending on how it goes
    if args.decompose_actions:
      transition_model = binary_classifier.BinaryClassifier(num_gen_features, 
        feature_size, args.hidden_size, args.cuda) 
      direction_model = binary_classifier.BinaryClassifier(num_features, 
        feature_size, args.hidden_size, args.cuda) 
    else:
      transition_model = classifier.Classifier(num_features, feature_size, 
        args.hidden_size, num_transitions, args.cuda) 
    if args.predict_relations:
      relation_model = classifier.Classifier(num_features, feature_size, 
          args.hidden_size, num_relations, args.cuda)
    else:
      relation_model = None
    if args.generative:
      word_model = classifier.Classifier(num_gen_features, feature_size,
              args.hidden_size, vocab_size, args.cuda)

    if args.cuda:
      encoder_model.cuda()
      transition_model.cuda()
      if args.predict_relations:
        relation_model.cuda()
      if args.generative:
        word_model.cuda()
      if args.decompose_actions:
        direction_model.cuda()
    criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.BCELoss()

    params = (list(encoder_model.parameters()) 
              + list(transition_model.parameters()))
    if args.predict_relations:
      params += list(relation_model.parameters())
    if args.generative:
      params += list(word_model.parameters())
    if args.decompose_actions:
      params += list(direction_model.parameters())

    optimizer = optim.Adam(params, lr=args.lr)
   
    prev_val_loss = None
    for epoch in range(1, args.epochs+1):
      print('Start training epoch %d' % epoch)
      epoch_start_time = time.time()
      
      random.shuffle(sentences)
      sentences.sort(key=len) 
      #TODO batch for RNN encoder

      total_loss = 0 
      encoder_model.train()

      for i, train_sent in enumerate(sentences):
        # Training loop
        start_time = time.time()

        # sentence encoder
        sentence_data = get_sentence_batch(train_sent, args.cuda)
        encoder_model.zero_grad()
        transition_model.zero_grad()
        if args.predict_relations:
          relation_model.zero_grad()
        if args.generative:
          word_model.zero_grad()
        normalize = nn.Sigmoid()

        encoder_state = encoder_model.init_hidden(batch_size)
        encoder_output = encoder_model(sentence_data, encoder_state)

        actions, words, labels, features, gen_features = train_oracle(train_sent.conll, 
                encoder_output, num_features==4)
        
        if args.decompose_actions:
          actions, directions = decompose_transitions(actions)

        # when will the direction logits be none? -> from features
        # but 2-features will be used for both sh and re

        #TODO next step in efficiency is to call classifier models after
        # features are concatenated into single tensor, to increase ||
        # Filter out Nones and concatenate to get training examples
        if args.decompose_actions:
          transition_logits = [transition_model(feat) if feat is not None
                               else None for feat in gen_features] 
          direction_logits = [direction_model(feat) 
                               if feat is not None and direct is not None
                             else None for feat, direct in zip(features,
                                 directions)] 
          # somehow logits are too big
          direction_output, dir_var = filter_logits(direction_logits,
              directions, float_var=True)
          transition_output, action_var = filter_logits(transition_logits,
              actions, float_var=True)
          #print(direction_output)
          #print(dir_var) 
        else:
          transition_logits = [transition_model(feat) if feat is not None
                               else None for feat in features] 
          transition_output, action_var = filter_logits(transition_logits,
              actions)
        
        if args.predict_relations:
          relation_logits = []
          for feat, action in zip(features, actions):
            if action != data_utils._SH and feat is not None:
              relation_logits.append(relation_model(feat))
            else:
              relation_logits.append(None)

          relation_output, label_var = filter_logits(relation_logits, labels)

        if args.generative:
          gen_word_logits = []
          for feat, action, word in zip(gen_features, actions, words):
            if action == data_utils._SH:
              assert word >= 0
              gen_word_logits.append(word_model(feat))
            else:
              gen_word_logits.append(None)
          gen_word_output, word_var = filter_logits(gen_word_logits, words)

        # at least for generative model, think carefully about loss scaling
        loss = None
        if args.decompose_actions:
          if transition_output is not None:
            #TODO confirm dimentions, format of binary classes
            #TODO loss is None
            # now both is None
            #print(transition_output)
            #print(action_var) 
            loss = binary_criterion(normalize(transition_output.view(-1)),
                           action_var)
            #print('trans loss')
            #print(loss)
            #print('trans loss {:5.4f}'.format(loss))
            if direction_logits is not None:
              #print(direction_output)
              #print(dir_var) 
              dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                           dir_var)
              #print('dir loss')
              #print(dir_loss)
              #print('dir loss {:5.4f}'.format(dir_loss))
              loss = loss + dir_loss if loss is not None else dir_loss

        else:
          if transition_output is not None:
            loss = criterion(transition_output.view(-1, num_transitions),
                           action_var)
         
        if args.predict_relations and relation_output is not None:
            rel_loss = criterion(relation_output.view(-1, num_relations), 
                                 label_var)
            loss = loss + rel_loss if loss is not None else rel_loss

        if args.generative and gen_word_output is not None:
            word_loss = criterion(gen_word_output.view(-1, vocab_size),
                                  word_var)
            loss = loss + word_loss if loss is not None else word_loss

        if loss is not None:
          loss.backward()
          if args.grad_clip > 0:
           nn_utils.clip_grad_norm(params, args.grad_clip)
           optimizer.step() 
          total_loss += loss.data
 
        if i % args.logging_interval == 0 and i > 0:
          cur_loss = total_loss[0] / args.logging_interval
          elapsed = time.time() - start_time
          print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(  # | ppl {:8.2f}
              epoch, i, len(sentences), 
              elapsed * 1000 / args.logging_interval, cur_loss))
              #math.exp(cur_loss)))
          total_loss = 0
          start_time = time.time()

      val_batch_size = 1
      total_loss = 0
      conll_predicted = []

      encoder_model.eval()
      normalize = nn.Sigmoid() 
      for val_sent in dev_sentences:
        sentence_data = get_sentence_batch(val_sent, args.cuda, evaluation=True)
        encoder_state = encoder_model.init_hidden(val_batch_size)
        encoder_output = encoder_model(sentence_data, encoder_state)
    
        #TODO evaluate word prediction for generative model
        #TODO transition logits for decomposed model
        if args.decompose_actions:
          # TODO Testing
          viterbi_decode(val_sent.conll, encoder_output, transition_model,
                  word_model, direction_model, relation_model)

          predict, transition_logits, direction_logits, actions, relation_logits, labels = greedy_decode(
            val_sent.conll, encoder_output, transition_model, relation_model,
            direction_model)
        else:
          predict, transition_logits, _, actions, relation_logits, labels = greedy_decode(
            val_sent.conll, encoder_output, transition_model, relation_model,
            more_context=(num_features==4))

        for j, token in enumerate(predict):
          # Convert labels to str
          if j > 0 and relation_model is not None and token.pred_relation_ind >= 0:
            predict[j].pred_relation = rel_vocab.get_word(token.pred_relation_ind)
        conll_predicted.append(predict) 

        # Filter out Nones to get examples for loss, then concatenate
        if args.decompose_actions:
          transition_output, action_var = filter_logits(transition_logits,
              actions, float_var=True)
          direction_output, dir_var = filter_logits(direction_logits,
            directions, float_var=True)
        else:
          transition_output, action_var = filter_logits(transition_logits,
             actions)

        if transition_output is not None:
          if args.decompose_actions:
            tr_loss = binary_criterion(normalize(transition_output.view(-1)),
                                    action_var).data
            #if math.isnan(tr_loss[0]):
            #  print('Transition loss')
            #  print(transition_output)
            #  print(action_var)
            total_loss += tr_loss
            if direction_output is not None:
              dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                                      dir_var).data
              #if math.isnan(dir_loss[0]):
              #  print('Direction loss')
              #  print(direction_output)
              #  print(dir_var)
              total_loss += dir_loss
          else:
            total_loss += criterion(transition_output.view(-1, num_transitions),
                                    action_var).data

          if args.predict_relations:
            relation_output, label_var = filter_logits(relation_logits, labels)
            if relation_output is not None:
              total_loss += criterion(relation_output.view(-1, num_relations),
                                      label_var).data

      data_utils.write_conll(working_path + args.dev_name + '.' + str(epoch) + '.output.conll', conll_predicted)
      val_loss = total_loss[0] / len(dev_sentences)
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(
          epoch, (time.time() - epoch_start_time), val_loss))
            #'valid ppl {:8.2f}'
      print('-' * 89)

      # save the model
      if args.save_model != '':
        model_fn = working_path + args.save_model + '_encoder.pt'
        with open(model_fn, 'wb') as f:
          torch.save(encoder_model, f)
        model_fn = working_path + args.save_model + '_transition.pt'
        with open(model_fn, 'wb') as f:
          torch.save(transition_model, f)
        if relation_model is not None:
          model_fn = working_path + args.save_model + '_relation.pt'
          with open(model_fn, 'wb') as f:
            torch.save(relation_model, f)
        if args.generative:
          model_fn = working_path + args.save_model + '_word.pt'
          with open(model_fn, 'wb') as f:
            torch.save(word_model, f)
        if args.decompose_actions:
          model_fn = working_path + args.save_model + '_direction.pt'
          with open(model_fn, 'wb') as f:
            torch.save(direction_model, f)
  train()
