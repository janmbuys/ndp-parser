# Author: Jan Buys

import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

import data_utils
import nn_utils
import arc_hybrid
import arc_eager 

def training_decode(args, tr_system, val_sentences, rel_vocab, epoch=-1):
  val_batch_size = 1
  total_loss = 0
  total_length = 0
  total_length_more = 0
  conll_predicted = []

  criterion = nn.CrossEntropyLoss(size_average=False)
  binary_criterion = nn.BCELoss(size_average=False)

  tr_system.encoder_model.eval()
  normalize = nn.Sigmoid() 

  print('Decoding dev sentences')
  for val_sent in val_sentences:
    sentence_data = nn_utils.get_sentence_data_batch([val_sent], args.cuda, evaluation=True)
    encoder_state = tr_system.encoder_model.init_hidden(val_batch_size)
    encoder_output = tr_system.encoder_model(sentence_data, encoder_state)

    if args.viterbi_decode:
      #score = tr_system.inside_score(val_sent.conll, encoder_output)
      predict, transition_logits, direction_logits, actions, relation_logits, labels, gen_word_logits, words = tr_system.viterbi_decode(
          val_sent.conll, encoder_output)
    else:
      predict, transition_logits, direction_logits, actions, relation_logits, labels, gen_word_logits, words = tr_system.greedy_decode(
          val_sent.conll, encoder_output)

    for j, token in enumerate(predict):
      # Convert labels to str
      if j > 0 and tr_system.relation_model is not None and token.pred_relation_ind >= 0:
        predict[j].pred_relation = rel_vocab.get_word(token.pred_relation_ind)
    conll_predicted.append(predict) 

    if args.decompose_actions:
      actions, directions = tr_system.decompose_transitions(actions)

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
    total_length_more += len(val_sent) 

    if args.generative:
      gen_word_output, word_var = nn_utils.filter_logits(gen_word_logits, words, use_cuda=args.cuda)
      if gen_word_output is not None:
        word_loss = criterion(gen_word_output.view(-1, tr_system.vocab_size),
                              word_var).data
        total_loss += word_loss

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
        total_loss += criterion(transition_output.view(-1, 
          tr_system.num_transitions), action_var).data

      if args.predict_relations:
        relation_output, label_var = nn_utils.filter_logits(relation_logits, labels, use_cuda=args.cuda)
        if relation_output is not None:
          total_loss += criterion(relation_output.view(-1, tr_system.num_relations),
                                  label_var).data

  working_path = args.working_dir + '/'
  file_id = '.' + str(epoch) if epoch >= 0 else ''
  data_utils.write_conll(working_path + args.dev_name + file_id + '.output.conll', conll_predicted)
  return total_loss, total_length, total_length_more


def training_score(args, tr_system, val_sentences):
  val_batch_size = 1
  total_loss = 0
  total_length = 0
  total_length_more = 0

  tr_system.encoder_model.eval()
  normalize = nn.Sigmoid() 
  if not args.generative:
    word_model = None
  
  print('Scoring val sentences')
  for val_sent in val_sentences:
    sentence_loss = 0

    sentence_data = nn_utils.get_sentence_data_batch([val_sent], args.cuda, evaluation=True)
    encoder_state = tr_system.encoder_model.init_hidden(val_batch_size)
    encoder_output = tr_system.encoder_model(sentence_data, encoder_state)
    total_length += len(val_sent) - 1 
    total_length_more += len(val_sent) 

    score = tr_system.inside_score(val_sent.conll, encoder_output)
    total_loss += score

  return total_loss, total_length, total_length_more

def decode(args, val_sentences, word_vocab, pos_vocab, rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  batch_size = 1
  model_path = args.working_dir + '/' + args.save_model
  non_lin = args.non_lin #TODO better interface

  # Build the model
  assert args.arc_hybrid or args.arc_eager
  if args.arc_hybrid:
    tr_system = arc_hybrid.ArcHybridTransitionSystem(vocab_size,
        num_relations, args.embedding_size, 
        args.hidden_size, args.num_layers, args.dropout,
        args.init_weight_range, args.bidirectional, 
        args.use_more_features, non_lin,
        args.predict_relations, args.generative,
        args.decompose_actions, args.stack_next,
        args.batch_size, args.cuda, model_path, True)
  elif args.arc_eager:
    tr_system = arc_eager.ArcEagerTransitionSystem(vocab_size,
        num_relations, args.embedding_size, args.hidden_size, args.num_layers,
        args.dropout, args.init_weight_range, args.bidirectional, 
        args.use_more_features, non_lin,
        args.predict_relations, args.generative,
        args.decompose_actions, args.stack_next,
        args.batch_size, args.cuda, model_path, True, args.late_reduce_oracle)

  print('Done loading models')

  decode_start_time = time.time()
  total_loss, total_length, total_length_more = training_decode(args,
          tr_system, val_sentences, rel_vocab)

  val_loss = total_loss[0] / total_length
  val_loss_more = total_loss[0] / total_length_more

  print('-' * 89)
  print('| decoding time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
  print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
       val_loss_more, math.exp(val_loss_more)))
  print('-' * 89)

def score(args, val_sentences, word_vocab, pos_vocab, rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  batch_size = 1
  model_path = args.working_dir + '/' + args.save_model
  assert args.generative
  non_lin = args.non_lin #TODO better interface

  # Build the model
  assert args.arc_hybrid or args.arc_eager
  if args.arc_hybrid:
    tr_system = arc_hybrid.ArcHybridTransitionSystem(vocab_size,
        num_relations, args.embedding_size, 
        args.hidden_size, args.num_layers, args.dropout,
        args.init_weight_range, args.bidirectional, 
        args.use_more_features, non_lin,
        args.predict_relations, args.generative,
        args.decompose_actions, args.stack_next, args.batch_size, args.cuda, model_path, True)
  elif args.arc_eager:
    tr_system = arc_eager.ArcEagerTransitionSystem(vocab_size,
        num_relations, args.embedding_size, args.hidden_size, args.num_layers,
        args.dropout, args.init_weight_range, args.bidirectional, 
        args.use_more_features, non_lin,
        args.predict_relations, args.generative,
        args.decompose_actions, args.stack_next, args.batch_size, args.cuda, model_path, True,
        args.late_reduce_oracle)

  print('Done loading models')

  decode_start_time = time.time()
  total_loss, total_length, total_length_more = training_score(args,
          tr_system, val_sentences)

  val_loss = - total_loss / total_length
  val_loss_more = - total_loss / total_length_more

  print('-' * 89)
  print('| scoring time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
  print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
       val_loss_more, math.exp(val_loss_more)))
  print('-' * 89)


def train(args, sentences, dev_sentences, word_vocab, pos_vocab, rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  model_path = args.working_dir + '/' + args.save_model
  lr = args.lr
  non_lin = args.non_lin #TODO better interface

  # Build the model
  assert args.arc_hybrid or args.arc_eager
  if args.arc_hybrid:
    tr_system = arc_hybrid.ArcHybridTransitionSystem(vocab_size,
        num_relations, args.embedding_size, 
        args.hidden_size, args.num_layers, args.dropout,
        args.init_weight_range, args.bidirectional, 
        args.use_more_features, non_lin,
        args.predict_relations, args.generative,
        args.decompose_actions, args.stack_next, args.batch_size, args.cuda,
        model_path, False)
  elif args.arc_eager:
    tr_system = arc_eager.ArcEagerTransitionSystem(vocab_size,
        num_relations, args.embedding_size, args.hidden_size, args.num_layers,
        args.dropout, args.init_weight_range, args.bidirectional, 
        args.use_more_features, non_lin,
        args.predict_relations, args.generative,
        args.decompose_actions, args.stack_next, args.batch_size, args.cuda,
        model_path, False, args.late_reduce_oracle)

  criterion = nn.CrossEntropyLoss(size_average=False)
  binary_criterion = nn.BCELoss(size_average=False)

  params = (list(tr_system.encoder_model.parameters()) 
            + list(tr_system.transition_model.parameters()))
  if args.predict_relations:
    params += list(tr_system.relation_model.parameters())
  if args.generative:
    params += list(tr_system.word_model.parameters())
  if args.decompose_actions:
    params += list(tr_system.direction_model.parameters())
  assert params is not None

  if args.adam:
    optimizer = optim.Adam(params, lr=lr)
  else:
    optimizer = optim.SGD(params, lr=lr)

  batch_size = 1
  prev_val_loss = None
  patience_count = 0
  for epoch in range(1, args.epochs+1):
    print('Start training epoch %d' % epoch)
    epoch_start_time = time.time()
    
    random.shuffle(sentences)
    sentences.sort(key=len) 

    total_loss = 0 
    global_loss = 0 
    total_num_tokens = 0 
    global_num_tokens = 0 
    tr_system.encoder_model.train()

    start_time = time.time()
    for i, train_sent in enumerate(sentences):
      # Training loop

      # sentence encoder
      sentence_data = nn_utils.get_sentence_data_batch([train_sent], args.cuda)
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
          train_sent.conll, encoder_output)
      
      if args.decompose_actions:
        stack_actions, directions = tr_system.decompose_transitions(actions)
      else:
        stack_actions = tr_system.map_transitions(actions)

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
            stack_actions, use_cuda=args.cuda)
      
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
          loss = criterion(transition_output.view(-1, tr_system.num_transitions),
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
        if args.grad_clip > 0: #TODO official gradient clipping
           nn.utils.clip_grad_norm(params, args.grad_clip) #TODO no gradient
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

    if True: # TODO maybe parameterize train decode option
      decode_start_time = time.time()
      total_loss, total_length, total_length_more = training_decode(args, 
          tr_system, dev_sentences, rel_vocab, epoch)
      val_loss = total_loss[0] / total_length
      val_loss_more = total_loss[0] / total_length_more

      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | {:5d} tokens | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
          epoch, (time.time() - epoch_start_time), total_length, val_loss,
          math.exp(val_loss)))
      print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
          val_loss_more, math.exp(val_loss_more)))
      print('-' * 89)
      print('decoding time: {:5.2f}s'.format(time.time() - decode_start_time))

    if args.generative:
      # score the model 
      decode_start_time = time.time()
      total_loss, total_length, total_length_more = training_score(args,
          tr_system, dev_sentences)

      val_loss = - total_loss / total_length
      val_loss_more = - total_loss / total_length_more

      print('-' * 89)
      print('| scoring time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
           (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
      print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
           val_loss_more, math.exp(val_loss_more)))
      print('-' * 89)

    # Anneal the learning rate.
    if (not args.adam and args.num_init_lr_epochs > 0 
        and epoch >= args.num_init_lr_epochs):
      if args.reduce_lr and val_loss > prev_val_loss:
        lr /= 2
      else:
        lr = lr / args.lr_decay
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if prev_val_loss and val_loss > prev_val_loss:
      patience_count += 1
    else:
      patience_count = 0
      prev_val_loss = val_loss
      store_this_iter = True
      # Saves the model. #TODO save only parameters
      if args.save_model != '':
        tr_system.store_model(args.working_dir + '/' + args.save_model)

    if args.store_all_iterations:
      tr_system.store_model(args.working_dir + '/' + args.save_model + '_' +
          str(epoch))

    if args.patience > 0 and patience_count >= args.patience:
      break


