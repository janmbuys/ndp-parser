# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source

import argparse
import os
import random
import sys
import time

from pathlib import Path

import torch

import data_utils

import supervised_parser
import unsupervised_parser

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
  parser.add_argument('--test', action='store_true', 
                      help='Evaluate test set', 
                      default=False)
  parser.add_argument('--decode_at_checkpoints', action='store_true', 
                      help='Decode at training checkpoints', 
                      default=False)

  parser.add_argument('--viterbi_decode', action='store_true',
                      help='Perform Viterbi decoding')
  parser.add_argument('--inside_decode', action='store_true',
                      help='Compute inside score for decoding')
  parser.add_argument('--arc_hybrid', action='store_true',
                      help='Arc hybrid transition system')
  parser.add_argument('--arc_eager', action='store_true',
                      help='Arc eager transition system')
  parser.add_argument('--late_reduce_oracle', action='store_true',
                      help='Arc eager late reduce oracle')

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
  parser.add_argument('--num_init_lr_epochs', type=int, default=-1, 
                      help='number of epochs before learning rate decay')
  parser.add_argument('--patience', type=int, default=10, 
                      help='Stop training if not improving for some number of epochs')
  parser.add_argument('--lr_decay', type=float, default=1.1,
                      help='learning rate decay per epoch')
  parser.add_argument('--tie_weights', action='store_true',
                      help='tie the word embedding and softmax weights')
  parser.add_argument('--reduce_lr', action='store_true',
                      help='reduce lr if val ppl does not improve')
  parser.add_argument('--init_weight_range', type=float, default=0.1,
                      help='weight initialization range')
  parser.add_argument('--adam', action='store_true',
                      help='use Adam optimizer')
  parser.add_argument('--cudnn', action='store_true', default=False)

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
  parser.add_argument('--stack_next', action='store_true',
                      help='predict stack-next instead of buffer-next')

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
  parser.add_argument('--store_all_iterations', action='store_true',
                      help='store model at all iterations')
  parser.add_argument('--save_model', type=str,  default='model',
                      help='path to save the final model, exclude extension')

  args = parser.parse_args()
  assert not (args.generative and args.bidirectional), 'Bidirectional encoder invalid for generative model'
  assert args.arc_hybrid or args.arc_eager or args.unsup
  assert not (args.use_more_features and args.decompose_actions), 'For decomposed features use small contexts'

  # TODO check if this has the same effect across different files
  torch.manual_seed(args.seed) 
  random.seed(args.seed)
  if args.cuda:
    assert torch.cuda.is_available(), 'Cuda not available!'
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.cudnn

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
        replicate_rnng=args.replicate_rnng_data)

  test_sentences, _,  _, _ = data_utils.read_sentences_given_vocab(
        data_path, args.test_name, data_working_path, projectify=False, 
        replicate_rnng=args.replicate_rnng_data)

  data_utils.write_conll_baseline(data_path + 'dev.baseline.conll', 
      [sent.conll for sent in dev_sentences])

  #data_utils.create_length_histogram(sentences, args.working_dir)

  if args.small_data:
    sentences = sentences[:500]
    #dev_sentences = dev_sentences
    dev_sentences = dev_sentences[:200]

  val_sentences = test_sentences if args.test else dev_sentences
  if args.unsup:
    if args.decode:          
      unsupervised_parser.decode(args, val_sentences, word_vocab)
    elif args.score:
      unsupervised_parser.score(args, val_sentences, word_vocab)
    else: 
      unsupervised_parser.train(args, sentences, val_sentences, word_vocab)
  else:
    if args.decode:          
      supervised_parser.decode(args, val_sentences, word_vocab, pos_vocab, rel_vocab)
    elif args.score:
      supervised_parser.score(args, val_sentences, word_vocab, pos_vocab, rel_vocab)
    else:
      supervised_parser.train(args, sentences, val_sentences, word_vocab, pos_vocab, rel_vocab)

