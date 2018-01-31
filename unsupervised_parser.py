# Author: Jan Buys
# Code credit: BIST parser; pytorch example word_language_model; 
#              pytorch master source; pytorch CRF example 

import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

import data_utils
import nn_utils
import shift_reduce_dp_cubic
import arc_eager_dp

def training_decode(val_sentences, stack_model, word_vocab, conll_output_fn,
        transition_output_fn, max_sents=-1, use_cuda=False):
  stack_model.eval()
  decode_start_time = time.time()
  greedy_loss = 0
  length = 0
  length_more = 0

  with open(conll_output_fn, 'w') as conll_fh:
    with open(transition_output_fn, 'w') as tr_fh:
      for i, val_sent in enumerate(val_sentences):
        if max_sents > 0 and i > max_sents:
          break
        sentence_data = nn_utils.get_sentence_data_batch([val_sent], use_cuda,
            evaluation=True)
        actions, dependents, greedy_word_loss = stack_model.forward(sentence_data)
        greedy_loss += greedy_word_loss
        length += len(val_sent) - 1 
        length_more += len(val_sent) 
        action_str = ' '.join([data_utils.transition_to_str(act) 
                               for act in actions])
        tr_fh.write(action_str + '\n')
        for i, entry in enumerate(val_sent.conll):
          if i > 0:
            pred_parent = dependents[i]
            #if stack_model.stack_next:
            #pred_parent = stack_shift_dependents[i]
            #else:
            #  pred_parent = buffer_shift_dependents[i]
            conll_fh.write('\t'.join([str(entry.id), entry.form, '_', 
              entry.cpos, entry.pos, 
              '_', str(pred_parent), '_', '_', '_']) + '\n')
        conll_fh.write('\n')
  print('decode time {:2.2f}s'.format(time.time() - decode_start_time))

  val_loss = - greedy_loss[0] / length
  val_loss_more = - greedy_loss[0] / length_more

  print('| greedy valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       val_loss, math.exp(val_loss)))
  print('       | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
       val_loss_more, math.exp(val_loss_more)))
  print('-' * 89)


def training_score(args, stack_model, val_sentences):
  val_batch_size = 1
  total_loss = 0
  total_length = 0
  total_length_more = 0

  stack_model.eval()
  normalize = nn.Sigmoid() 
  
  print('Scoring val sentences')
  for val_sent in val_sentences:
    sentence_data = nn_utils.get_sentence_data_batch([val_sent], args.cuda,
        evaluation=True)
    loss = stack_model.neg_log_likelihood(sentence_data)
    total_loss += loss.data
    total_length += len(val_sent) - 1 
    total_length_more += len(val_sent) 

  return total_loss, total_length, total_length_more


def decode(args, val_sentences, word_vocab, score=False):
  vocab_size = len(word_vocab)
  batch_size = 1
  model_path = args.working_dir + '/' + args.save_model
  #non_lin = args.non_lin #TODO better interface
  #gen_non_lin = args.gen_non_lin

  assert model_path != ''
  print('Loading models')
  model_fn = model_path + '_stack.pt'
  with open(model_fn, 'rb') as f:
    stack_model = torch.load(f)
 
  if args.cuda:
    stack_model.cuda()

  print('Done loading models')
  decode_start_time = time.time()

  if score:
    total_loss, total_length, total_length_more = training_score(args,
        stack_model, val_sentences)

    val_loss = total_loss[0] / total_length
    val_loss_more = total_loss[0] / total_length_more

    print('-' * 89)
    print('| decoding time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
         (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
    print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
         val_loss_more, math.exp(val_loss_more)))
    print('-' * 89)
  else: # not currently computing ppl here 
    out_name = args.working_dir + '/' + args.dev_name 
    training_decode(val_sentences, stack_model, word_vocab, out_name + '.output.conll',
        out_name + '.output.tr', use_cuda=args.cuda) 


def train(args, sentences, dev_sentences, word_vocab):
  vocab_size = len(word_vocab)
  batch_size = args.batch_size
  assert args.generative and not args.bidirectional
  
  # Build the model
  non_lin = args.non_lin
  gen_non_lin = args.gen_non_lin

  if args.arc_eager:
    stack_model = arc_eager_dp.ArcEagerDP(vocab_size, args.embedding_size,
      args.hidden_size, args.num_layers, args.dropout,
      args.init_weight_range, non_lin, gen_non_lin, args.decompose_actions, 
      args.stack_next, args.embed_only, args.embed_only_gen, 
      args.with_valency, args.cuda)
  else:
    stack_model = shift_reduce_dp_cubic.ShiftReduceDP(vocab_size, args.embedding_size,
      args.hidden_size, args.num_layers, args.dropout,
      args.init_weight_range, non_lin,
      gen_non_lin, args.stack_next, args.embed_only, args.embed_only_gen, args.cuda)

  if args.cuda:
    stack_model.cuda()

  lr = args.lr 
  params = list(stack_model.parameters()) 
  if args.adam:
    optimizer = optim.Adam(params, lr=lr)
  else:
    optimizer = optim.SGD(params, lr=lr)
 
  prev_val_loss = None
  patience_count = 0
  for epoch in range(1, args.epochs+1):
    print('Start unsup training epoch %d' % epoch)
    epoch_start_time = time.time()
    
    random.shuffle(sentences)
    sentences.sort(key=len) #, reverse=True) #temp 
    stack_model.train()

    total_loss = 0 
    global_loss = 0 
    total_num_tokens = 0 
    global_num_tokens = 0 
    batch_count = 0

    # new batching
    batch_inds = []
    i = 0  
    while i < len(sentences):
      # Training loop
      length = len(sentences[i])
      j = i + 1
      if args.token_balanced_batches:
        while (j < len(sentences) and len(sentences[j]) == length
               and (j - i)*length < batch_size):
          j += 1
      else:
        while (j < len(sentences) and len(sentences[j]) == length
             and (j - i) < batch_size):
          j += 1
      batch_inds.append(list(range(i, j)))
      i = j
    random.shuffle(batch_inds)

    start_time = time.time()
    print('Training size {:3d}'.format(len(sentences)))

    for batch_ind in batch_inds: 
      sentence_data = nn_utils.get_sentence_data_batch(
          [sentences[k] for k in batch_ind], args.cuda)
      local_batch_size = len(batch_ind)
      batch_count += 1

      stack_model.train()
      stack_model.zero_grad()
      loss = stack_model.neg_log_likelihood(sentence_data)
      #print(loss)

      if loss is not None:
        loss.backward()
        if args.grad_clip > 0:
          nn.utils.clip_grad_norm(params, args.grad_clip)
        optimizer.step() 
        total_loss += loss.data
        global_loss += loss.data

      batch_tokens = (sentence_data.size()[0] - 1)*local_batch_size
      total_num_tokens += batch_tokens
      global_num_tokens += batch_tokens

      if batch_count % args.logging_interval == 0 and i > 0:
        if args.decode_at_checkpoints:
          out_name = (args.working_dir + '/' + args.dev_name + '.' 
                      + str(epoch) + '.' + str(batch_count))
          training_decode(dev_sentences, stack_model, word_vocab, 
              out_name + '.conll', out_name + '.tr', -1, args.cuda)
          stack_model.train()

        cur_loss = total_loss[0] / total_num_tokens
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, batch_count, lr,
            elapsed * 1000 / args.logging_interval, cur_loss, 
            math.exp(cur_loss)))
        total_loss = 0
        total_num_tokens = 0
        start_time = time.time()

    avg_global_loss = global_loss[0] / global_num_tokens
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | {:5d} batches | tokens {:5d} | loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), batch_count, global_num_tokens,
        avg_global_loss, math.exp(avg_global_loss)))

    # score the model
    decode_start_time = time.time()
    total_loss, total_length, total_length_more = training_score(args,
        stack_model, dev_sentences)

    val_loss = total_loss[0] / total_length
    val_loss_more = total_loss[0] / total_length_more
    print('-' * 89)
    print('| eval time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
          (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
    print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(val_loss_more,
        math.exp(val_loss_more)))
    print('-' * 89)

    # Decoding
    out_name = args.working_dir + '/' + args.dev_name + '.' + str(epoch) 
    training_decode(dev_sentences, stack_model, word_vocab, out_name + '.output.conll',
        out_name + '.output.tr', use_cuda=args.cuda)
    
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
      # save the model
      if args.save_model != '':
        model_fn = args.working_dir + '/' + args.save_model + '_stack.pt'
        with open(model_fn, 'wb') as f:
          torch.save(stack_model, f)
    if args.patience > 0 and patience_count >= args.patience:
      break

