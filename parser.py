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
import dp_stack
import data_utils
import nn_utils
import arc_hybrid
import arc_eager 
import stack_parser

def train(args, sentences, dev_sentences, test_sentences, word_vocab, 
          pos_vocab, rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  num_transitions = 3
  num_features = 2 if args.decompose_actions else 4

  # Build the model
  assert args.arc_hybrid or args.arc_eager
  if args.arc_hybrid:
    tr_system = arc_hybrid_parser.ArcHybridTransitionSystem(vocab_size,
        num_relations, num_features, num_transitions, args.embedding_size, 
        args.hidden_size, args.num_layers, args.dropout, args.bidirectional, 
        args.use_more_features, args.predict_relations, args.generative,
        args.decompose_actions, args.batch_size, args.cuda)
  elif args.arc_eager:
    tr_system = arc_eager_parser.ArcEagerTransitionSystem(vocab_size,
        num_relations, num_features, num_transitions, args.embedding_size, 
        args.hidden_size, args.num_layers, args.dropout, args.bidirectional, 
        args.batch_size, args.cuda)

  criterion = nn.CrossEntropyLoss(size_average=args.criterion_size_average)
  binary_criterion = nn.BCELoss(size_average=args.criterion_size_average)

  params = (list(tr_system.encoder_model.parameters()) 
            + list(tr_system.transition_model.parameters()))
  if args.predict_relations:
    params += list(tr_system.relation_model.parameters())
  if args.generative:
    params += list(tr_system.word_model.parameters())
  if args.decompose_actions:
    params += list(tr_system.direction_model.parameters())

  optimizer = optim.Adam(params, lr=args.lr)
  batch_size = 1
 
  prev_val_loss = None
  for epoch in range(1, args.epochs+1):
    print('Start training epoch %d' % epoch)
    epoch_start_time = time.time()
    
    random.shuffle(sentences)
    #sentences.sort(key=len) 

    total_loss = 0 
    global_loss = 0 
    total_num_tokens = 0 
    global_num_tokens = 0 
    tr_system.encoder_model.train()

    start_time = time.time()
    for i, train_sent in enumerate(sentences):
      # Training loop
      print(i)

      # sentence encoder
      sentence_data = nn_utils.get_sentence_batch(train_sent, args.cuda)
      tr_system.encoder_model.zero_grad()
      tr_system.transition_model.zero_grad()
      if args.predict_relations:
        tr_system.relation_model.zero_grad()
      if args.generative:
        tr_system.word_model.zero_grad()
      normalize = nn.Sigmoid()

      encoder_state = tr_system.encoder_model.init_hidden(batch_size)
      encoder_output = tr_system.encoder_model(sentence_data, encoder_state)

      actions, words, labels, features = tr_system.oracle(
          train_sent.conll, encoder_output, args.use_more_features, 
          use_cuda=args.cuda)
      
      if args.decompose_actions:
        stack_actions, directions = tr_system.decompose_transitions(actions)

      # when will the direction logits be none? -> from features
      # but 2-features will be used for both sh and re

      # Filter out Nones and concatenate to get training examples
      if args.decompose_actions:
        transition_logits = [tr_system.transition_model(feat) if feat is not None
                             else None for feat in features] 
        direction_logits = [tr_system.direction_model(feat) 
                             if feat is not None and direct is not None
                           else None for feat, direct in zip(features,
                               directions)] 
        direction_output, dir_var = nn_utils.filter_logits(direction_logits,
            directions, float_var=True, use_cuda=args.cuda)
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
            stack_actions, float_var=True, use_cuda=args.cuda)
      else:
        transition_logits = [tr_system.transition_model(feat) if feat is not None
                             else None for feat in features] 
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
            actions, use_cuda=args.cuda)
      
      if args.predict_relations:
        relation_logits = []
        for feat, action in zip(features, actions):
          if ((action == data_utils._LA or action == data_utils._RA) 
              and feat is not None):
            relation_logits.append(tr_system.relation_model(feat))
          else:
            relation_logits.append(None)

        relation_output, label_var = nn_utils.filter_logits(relation_logits, labels, use_cuda=args.cuda)

      if args.generative:
        gen_word_logits = []
        for feat, action, word in zip(features, actions, words):
          if action in tr_system.generate_actions:
            assert word >= 0
            gen_word_logits.append(tr_system.word_model(feat))
          else:
            gen_word_logits.append(None)
        gen_word_output, word_var = nn_utils.filter_logits(gen_word_logits, words, use_cuda=args.cuda)

      loss = None
      if args.decompose_actions:
        if transition_output is not None:
          loss = binary_criterion(normalize(transition_output.view(-1)),
                         action_var)
          if direction_logits is not None:
            dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                         dir_var)
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

      total_num_tokens += len(train_sent) - 1 
      global_num_tokens += len(train_sent) - 1 
      if loss is not None:
        loss.backward()
        if args.grad_clip > 0:
          nn_utils.clip_grad_norm(params, args.grad_clip)
        optimizer.step() 
        total_loss += loss.data
        global_loss += loss.data

      if i % args.logging_interval == 0 and i > 0:
        cur_loss = total_loss[0] / total_num_tokens
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} '.format(   
            epoch, i, len(sentences), 
            elapsed * 1000 / args.logging_interval, cur_loss,
            math.exp(cur_loss)))
        total_loss = 0
        total_num_tokens = 0
        start_time = time.time()
   
    avg_global_loss = global_loss[0] / global_num_tokens
    print('-' * 89)
    print('| end of epoch {:3d} | {:5d} batches | tokens {:5d} | loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, len(sentences), global_num_tokens,
        avg_global_loss, math.exp(avg_global_loss)))

    #TODO move to a different method
    val_batch_size = 1
    total_loss = 0
    total_length = 0
    conll_predicted = []
    decode_start_time = time.time()

    tr_system.encoder_model.eval()
    normalize = nn.Sigmoid() 
    if not args.generative:
      word_model = None
  
    print('Decoding dev sentences')
    for val_sent in dev_sentences:
      sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda, evaluation=True)
      encoder_state = tr_system.encoder_model.init_hidden(val_batch_size)
      encoder_output = tr_system.encoder_model(sentence_data, encoder_state)
  
      if args.decompose_actions:
        if args.viterbi_decode: #TODO from here
          predict, transition_logits, direction_logits, actions, relation_logits, labels = viterbi_decode(
              val_sent.conll, encoder_output, transition_model, word_model,
              direction_model, relation_model)
          print(actions)
        else:
          predict, transition_logits, direction_logits, actions, relation_logits, labels = greedy_decode(
            val_sent.conll, encoder_output, transition_model, relation_model,
            direction_model, use_cuda=args.cuda)
      else:
        predict, transition_logits, _, actions, relation_logits, labels = greedy_decode(
          val_sent.conll, encoder_output, transition_model, relation_model,
          more_context=args.use_more_features, use_cuda=args.cuda)

      #TODO need to compute word probabilities here
      for j, token in enumerate(predict):
        # Convert labels to str
        if j > 0 and relation_model is not None and token.pred_relation_ind >= 0:
          predict[j].pred_relation = rel_vocab.get_word(token.pred_relation_ind)
      conll_predicted.append(predict) 

      if args.decompose_actions:
        actions, directions = decompose_transitions(actions)

      # Filter out Nones to get examples for loss, then concatenate
      if args.decompose_actions:
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
            actions, float_var=True, use_cuda=args.cuda)
        direction_output, dir_var = nn_utils.filter_logits(direction_logits,
          directions, float_var=True, use_cuda=args.cuda)
      else:
        transition_output, action_var = nn_utils.filter_logits(transition_logits,
           actions, use_cuda=args.cuda)

      total_length += len(val_sent) - 1
      if transition_output is not None:
        if args.decompose_actions:
          tr_loss = binary_criterion(normalize(transition_output.view(-1)),
                                  action_var).data
          total_loss += tr_loss
          if direction_output is not None:
            dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                                    dir_var).data
            total_loss += dir_loss
        else:
          total_loss += criterion(transition_output.view(-1, num_transitions),
                                  action_var).data

        if args.predict_relations:
          relation_output, label_var = nn_utils.filter_logits(relation_logits, labels, use_cuda=args.cuda)
          if relation_output is not None:
            total_loss += criterion(relation_output.view(-1, num_relations),
                                    label_var).data

    working_path = args.working_dir + '/'
    data_utils.write_conll(working_path + args.dev_name + '.' + str(epoch) + '.output.conll', conll_predicted)
    val_loss = total_loss[0] / total_length
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | {:5d} tokens | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
        epoch, (time.time() - epoch_start_time), total_length, val_loss,
        math.exp(val_loss)))
    print('decoding time: {:5.2f}s'.format(time.time() - decode_start_time))
    print('-' * 89)

    # save the model #TODO save only parameters
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


def score(args, dev_sentences, test_sentences, word_vocab, pos_vocab, 
          rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  num_transitions = 3
  num_features = 4 if args.use_more_features else 2
  batch_size = args.batch_size

  working_path = args.working_dir + '/'
  print('Loading models')
  # Load models. TODO only load parameters
  model_fn = working_path + args.save_model + '_encoder.pt'
  with open(model_fn, 'rb') as f:
    encoder_model = torch.load(f)
  model_fn = working_path + args.save_model + '_transition.pt'
  with open(model_fn, 'rb') as f:
    transition_model = torch.load(f)
  if args.predict_relations:
    model_fn = working_path + args.save_model + '_relation.pt'
    with open(model_fn, 'rb') as f:
      relation_model = torch.load(f)
  if args.generative:
    model_fn = working_path + args.save_model + '_word.pt'
    with open(model_fn, 'rb') as f:
      word_model = torch.load(f)
  if args.decompose_actions:
    model_fn = working_path + args.save_model + '_direction.pt'
    with open(model_fn, 'rb') as f:
      direction_model = torch.load(f)

  #TODO not sure if this is neccessary
  if args.cuda:
    encoder_model.cuda()
    transition_model.cuda()
    if args.predict_relations:
      relation_model.cuda()
    if args.generative:
      word_model.cuda()
    if args.decompose_actions:
      direction_model.cuda()

  criterion = nn.CrossEntropyLoss(size_average=args.criterion_size_average)
  binary_criterion = nn.BCELoss(size_average=args.criterion_size_average)

  print('Done loading models')

  val_batch_size = 1
  total_loss = 0
  total_length = 0
  conll_predicted = []
  dev_losses = []
  decode_start_time = time.time()

  encoder_model.eval()
  normalize = nn.Sigmoid() 
  if not args.generative:
    word_model = None

  for val_sent in dev_sentences:
    sentence_loss = 0
    sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda, evaluation=True)
    encoder_state = encoder_model.init_hidden(val_batch_size)
    encoder_output = encoder_model(sentence_data, encoder_state)
    total_length += len(val_sent) - 1 

    if args.decompose_actions and args.inside_decode: 
      inside_score = inside_score_decode(val_sent.conll, encoder_output,
            transition_model, word_model, use_cuda=args.cuda)
      #inside_score.backward()
      #score = nn_utils.to_numpy(inside_score)[0]  
      score = inside_score
      print(score)
      dev_losses.append(score)
      total_loss += score
    #else:
      #TODO greedy score

  val_loss = - total_loss / total_length
  print('-' * 89)
  print('| decoding time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
  print('-' * 89)


def decode(args, dev_sentences, test_sentences, word_vocab, pos_vocab, 
           rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  num_transitions = 3
  num_features = 4 if args.use_more_features else 2
  batch_size = args.batch_size

  working_path = args.working_dir + '/'
  print('Loading models')
  # Load models. TODO only load parameters
  model_fn = working_path + args.save_model + '_encoder.pt'
  with open(model_fn, 'rb') as f:
    encoder_model = torch.load(f)
  model_fn = working_path + args.save_model + '_transition.pt'
  with open(model_fn, 'rb') as f:
    transition_model = torch.load(f)
  if args.predict_relations:
    model_fn = working_path + args.save_model + '_relation.pt'
    with open(model_fn, 'rb') as f:
      relation_model = torch.load(f)
  if args.generative:
    model_fn = working_path + args.save_model + '_word.pt'
    with open(model_fn, 'rb') as f:
      word_model = torch.load(f)
  if args.decompose_actions:
    model_fn = working_path + args.save_model + '_direction.pt'
    with open(model_fn, 'rb') as f:
      direction_model = torch.load(f)

  #TODO not sure if this is neccessary
  if args.cuda:
    encoder_model.cuda()
    transition_model.cuda()
    if args.predict_relations:
      relation_model.cuda()
    if args.generative:
      word_model.cuda()
    if args.decompose_actions:
      direction_model.cuda()

  criterion = nn.CrossEntropyLoss(size_average=args.criterion_size_average)
  binary_criterion = nn.BCELoss(size_average=args.criterion_size_average)

  print('Done loading models')

  val_batch_size = 1
  total_loss = 0
  total_length = 0
  conll_predicted = []
  dev_losses = []
  decode_start_time = time.time()

  encoder_model.eval()
  normalize = nn.Sigmoid() 
  if not args.generative:
    word_model = None

  for val_sent in dev_sentences:
    sentence_loss = 0
    sentence_data = nn_utils.get_sentence_batch(val_sent, args.cuda, evaluation=True)
    encoder_state = encoder_model.init_hidden(val_batch_size)
    encoder_output = encoder_model(sentence_data, encoder_state)
    total_length += len(val_sent) - 1 

    #TODO evaluate word prediction for generative model
    if args.decompose_actions:
      if args.viterbi_decode:
        predict, transition_logits, direction_logits, actions, relation_logits, labels = viterbi_decode(
            val_sent.conll, encoder_output, transition_model, word_model,
            direction_model, relation_model, use_cuda=args.cuda)
        print(actions)
      else:
        predict, transition_logits, direction_logits, actions, relation_logits, labels = greedy_decode(
          val_sent.conll, encoder_output, transition_model, relation_model,
          direction_model, use_cuda=args.cuda)
    else:
      predict, transition_logits, _, actions, relation_logits, labels = greedy_decode(
        val_sent.conll, encoder_output, transition_model, relation_model,
        more_context=args.use_more_features, use_cuda=args.cuda)

    for j, token in enumerate(predict):
      # Convert labels to str
      if j > 0 and relation_model is not None and token.pred_relation_ind >= 0:
        predict[j].pred_relation = rel_vocab.get_word(token.pred_relation_ind)
    conll_predicted.append(predict) 

    if args.decompose_actions:
      actions, directions = decompose_transitions(actions)

    # Filter out Nones to get examples for loss, then concatenate
    if args.decompose_actions:
      transition_output, action_var = nn_utils.filter_logits(transition_logits,
          actions, float_var=True, use_cuda=args.cuda)
      direction_output, dir_var = nn_utils.filter_logits(direction_logits,
        directions, float_var=True, use_cuda=args.cuda)
    else:
      transition_output, action_var = nn_utils.filter_logits(transition_logits,
         actions, use_cuda=args.cuda)

    if transition_output is not None:
      if args.decompose_actions:
        tr_loss = binary_criterion(normalize(transition_output.view(-1)),
                                action_var).data
        #if math.isnan(tr_loss[0]):
        #  print('Transition loss')
        #  print(transition_output)
        #  print(action_var)
        sentence_loss += tr_loss
        if direction_output is not None:
          dir_loss = binary_criterion(normalize(direction_output.view(-1)),
                                  dir_var).data
          #if math.isnan(dir_loss[0]):
          #  print('Direction loss')
          #  print(direction_output)
          #  print(dir_var)
          sentence_loss += dir_loss
      else:
        sentence_loss += criterion(transition_output.view(-1, num_transitions),
                                action_var).data

      if args.predict_relations:
        relation_output, label_var = nn_utils.filter_logits(relation_logits,
            labels, use_cuda=args.cuda)
        if relation_output is not None:
          sentence_loss += criterion(relation_output.view(-1, num_relations),
                                  label_var).data
    total_loss += sentence_loss
    dev_losses.append(sentence_loss[0])

  with open(working_path + args.dev_name + '.score', 'w') as fh:
    for loss in dev_losses:
      fh.write(str(loss) + '\n')
  data_utils.write_conll(working_path + args.dev_name + '.output.conll', conll_predicted)
  val_loss = total_loss[0] / total_length
  print('-' * 89)
  print('| decoding time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
  print('-' * 89)


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
  parser.add_argument('--decode', action='store_true', 
                      help='Only decode, assuming existing model', 
                      default=False)
  parser.add_argument('--score', action='store_true', 
                      help='Only score, assuming existing model', 
                      default=False)
  parser.add_argument('--viterbi_decode', action='store_true',
                      help='Perform Viterbi decoding')
  parser.add_argument('--inside_decode', action='store_true',
                      help='Compute inside score for decoding')
  parser.add_argument('--arc_hybrid', action='store_true',
                      help='Arc hybrid transition system')
  parser.add_argument('--arc_eager', action='store_true',
                      help='Arc eager transition system')

  parser.add_argument('--embedding_size', type=int, default=128,
                      help='size of word embeddings')
  parser.add_argument('--hidden_size', type=int, default=128, 
                      help='humber of hidden units per layer')
  parser.add_argument('--num_layers', type=int, default=1,
                      help='number of layers')
  parser.add_argument('--lr', type=float, default=0.001,
                      help='initial learning rate')
  parser.add_argument('--grad_clip', type=float, default=5,
                      help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=100,
                      help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                      help='batch size')
  parser.add_argument('--dropout', type=float, default=0.0, 
                      help='dropout rate')
  parser.add_argument('--bidirectional', action='store_true',
                      help='use bidirectional encoder')
  parser.add_argument('--generative', action='store_true',
                      help='use generative parser')
  parser.add_argument('--decompose_actions', action='store_true',
                      help='decompose action prediction to fit dynamic program')
  parser.add_argument('--use_more_features', action='store_true',
                      help='use 4 instead of 2 features')
  parser.add_argument('--unsup', action='store_true',
                      help='unsupervised model')

  parser.add_argument('--criterion_size_average', 
                      dest='criterion_size_average', action='store_true', 
                      help='Global loss averaging')
  parser.add_argument('--no_criterion_size_average',
                      dest='criterion_size_average', action='store_true', 
                      help='Local loss averaging')
  parser.set_defaults(criterion_size_average=True)
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
  parser.add_argument('--max_sentence_length', type=int, default=-1,
                      help='maximum training sentence length')
  parser.add_argument('--small_data', action='store_true',
                      help='use small version of dataset')
  parser.add_argument('--logging_interval', type=int, default=5000, 
                      metavar='N', help='report ppl every x steps')
  parser.add_argument('--save_model', type=str,  default='model.pt',
                      help='path to save the final model')

  args = parser.parse_args()
  assert not (args.generative and args.bidirectional), 'Bidirectional encoder invalid for generative model'
  assert args.arc_hybrid or args.arc_eager or args.unsup
  if args.viterbi_decode or args.inside_decode:
    assert args.decompose_actions, 'Decomposed actions required for dynamic programming'
  assert not (args.use_more_features and args.decompose_actions), 'For decomposed features use small contexts'

  # TODO check if this has the same effect across different files
  torch.manual_seed(args.seed) 
  random.seed(args.seed)
  if args.cuda:
    assert torch.cuda.is_available(), 'Cuda not available.'
    torch.cuda.manual_seed(args.seed)

  data_path = args.data_dir + '/' 
  data_working_path = args.data_working_dir + '/'

  # Prepare training data
  vocab_path = Path(data_working_path + 'vocab')
  if vocab_path.is_file() and not args.reset_vocab:
    sentences, word_vocab, pos_vocab, rel_vocab = data_utils.read_sentences_given_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data,
        max_length=args.max_sentence_length)
  else:     
    print('Preparing vocab')
    sentences, word_vocab, pos_vocab, rel_vocab = data_utils.read_sentences_create_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        replicate_rnng=args.replicate_rnng_data, 
        max_length=args.max_sentence_length)

  # Read dev and test files with given vocab
  dev_sentences, _, _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.dev_name, data_working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data,
        max_length=args.max_sentence_length)

  test_sentences, _,  _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.test_name, data_working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data,
        max_length=args.max_sentence_length)

  data_utils.create_length_histogram(sentences)

  if args.small_data:
    sentences = sentences[:500]
    dev_sentences = dev_sentences[:100]

  if args.decode:          
    decode(args, dev_sentences, test_sentences, word_vocab, 
            pos_vocab, rel_vocab)
  elif args.score:
    score(args, dev_sentences, test_sentences, word_vocab, 
            pos_vocab, rel_vocab)
  elif args.unsup:
    stack_parser.train_unsup(args, sentences, dev_sentences, test_sentences, 
        word_vocab)
  else:
    train(args, sentences, dev_sentences, test_sentences, word_vocab, 
          pos_vocab, rel_vocab)

