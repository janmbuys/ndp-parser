# Author: Jan Buys
# Code credit: BIST parser

import argparse

import os
import sys
import time

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
  parser.add_argument('--test_file',
                      help='Raw text file to parse with trained model', 
                      metavar='FILE')
  parser.add_argument('--replicate_rnng_data', action='store_true', 
                      default=False)

  args = parser.parse_args()

  print('Preparing vocab')
  train_filename = args.data_dir + '/' + args.train_name + '.conll'
  word_counts, form_vocab, w2i, pos, rels = utils.vocab(train_filename,
      replicate_rnng=args.replicate_rnng_data)

  utils.write_vocab(args.working_dir + '/vocab', word_counts)
  



