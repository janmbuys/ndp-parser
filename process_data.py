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

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True,
                      help='Directory of Annotated CONLL files')
  parser.add_argument('--data_working_dir', required=True,
                      help='Working directory for data files')
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
  parser.add_argument('--swap_subject', action='store_true', 
                      default=False)
  parser.add_argument('--max_sentence_length', type=int, default=-1,
                      help='maximum training sentence length')
  parser.add_argument('--no_unk_classes', action='store_true', 
                      default=False)

  args = parser.parse_args()

  data_path = args.data_dir + '/' 
  data_working_path = args.data_working_dir + '/'

  # Prepare training data
  vocab_path = Path(data_working_path + 'vocab')
  if vocab_path.is_file() and not args.reset_vocab:
    sentences, word_vocab, pos_vocab, rel_vocab = data_utils.read_sentences_given_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        use_unk_classes = not args.no_unk_classes,
        replicate_rnng=args.replicate_rnng_data,
        max_length=args.max_sentence_length, swap_subject=args.swap_subject)
  else:     
    print('Preparing vocab')
    sentences, word_vocab, pos_vocab, rel_vocab = data_utils.read_sentences_create_vocab(
        data_path, args.train_name, data_working_path, projectify=True, 
        use_unk_classes = not args.no_unk_classes,
        replicate_rnng=args.replicate_rnng_data, 
        max_length=args.max_sentence_length, swap_subject=args.swap_subject )

    data_utils.create_length_histogram(sentences, data_working_path)

  # Read dev and test files with given vocab
  dev_sentences, _, _, _ = data_utils.read_sentences_given_vocab(
      data_path, args.dev_name, data_working_path, projectify=False, 
      use_unk_classes = not args.no_unk_classes,
      replicate_rnng=args.replicate_rnng_data, swap_subject=args.swap_subject)

  test_sentences, _,  _, _ = data_utils.read_sentences_given_vocab(
      data_path, args.test_name, data_working_path, projectify=False, 
      use_unk_classes = not args.no_unk_classes,
      replicate_rnng=args.replicate_rnng_data, swap_subject=args.swap_subject)

  data_utils.write_conll_baseline(data_working_path + 'dev.baseline.conll', 
      [sent.conll for sent in dev_sentences])


