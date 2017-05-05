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
import arc_hybrid_parser
import stack_parser

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

  if args.small_data:
    sentences = sentences[:10]
    dev_sentences = dev_sentences[:10]

  #data_utils.create_length_histogram(sentences)

  # Extract static oracle sequences
  #for sent in sentences:
  #  actions, labels = data_utils.oracle(sent.conll)

  if args.decode:          
    arg_hybrid_parser.decode(args, dev_sentences, test_sentences, word_vocab, 
            pos_vocab, rel_vocab)
  elif args.score:
    arg_hybrid_parser.score(args, dev_sentences, test_sentences, word_vocab, 
            pos_vocab, rel_vocab)
  elif args.unsup:
    stack_parser.train_unsup(args, sentences, dev_sentences, test_sentences, 
        word_vocab)
  else:
    #TODO add arc eager
    arc_hybrid_parser.train(args, sentences, dev_sentences, test_sentences, 
        word_vocab, pos_vocab, rel_vocab)

