# Author: Jan Buys
# Code credit: BIST parser

import argparse

import os
import sys
import time
from pathlib import Path

import utils

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True,
                      help='Directory of Annotated CONLL files')
  parser.add_argument('--working_dir', required=True,
                      help='Working directory for parser')
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

  args = parser.parse_args()
  
  # For training

  working_path = args.working_dir + '/'
  data_path = args.data_dir + '/' 
  print('Preparing vocab')
  vocab_path = Path(working_path + 'vocab')
  if vocab_path.is_file():
    #TODO read pos, rel vocab lists
    conll_sentences, word_counts, form_vocab, w2i = utils.read_sentences_given_vocab(
        data_path, args.train_name, working_path, replicate_rnng=args.replicate_rnng_data)
  else:     
    conll_sentences, word_counts, form_vocab, w2i, pos, rels = utils.read_sentences_create_vocab(
        data_path, args.train_name, working_path, replicate_rnng=args.replicate_rnng_data)

  # Read dev and test files with given vocab
  conll_dev_sentences, _, _, _ = utils.read_sentences_given_vocab(
        data_path, args.dev_name, working_path, replicate_rnng=args.replicate_rnng_data)

  conll_test_sentences, _, _, _ = utils.read_sentences_given_vocab(
        data_path, args.test_name, working_path, replicate_rnng=args.replicate_rnng_data)



