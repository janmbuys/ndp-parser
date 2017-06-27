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
import shift_reduce_dp
import arc_eager_dp

def training_decode(val_sentences, stack_model, word_vocab, conll_output_fn,
        transition_output_fn, max_sents=-1, use_cuda=False):
  stack_model.eval()
  decode_start_time = time.time()
  with open(conll_output_fn, 'w') as conll_fh:
    with open(transition_output_fn, 'w') as tr_fh:
      for i, val_sent in enumerate(val_sentences):
        if max_sents > 0 and i > max_sents:
          break
        sentence_data = nn_utils.get_sentence_data_batch([val_sent], use_cuda,
            evaluation=True)
        transition_logits, actions, dependents = stack_model.forward(sentence_data)
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
 

def train(args, sentences, dev_sentences, word_vocab):
  vocab_size = len(word_vocab)
  num_features = 2
  batch_size = args.batch_size
  assert args.generative and not args.bidirectional
  
  # Build the model
  feature_size = args.hidden_size
  non_lin = args.non_lin
  gen_non_lin = args.gen_non_lin

  if args.arc_eager:
    stack_model = arc_eager_dp.ArcEagerDP(vocab_size, args.embedding_size,
      args.hidden_size, args.num_layers, args.dropout,
      args.init_weight_range, num_features, args.stack_next, non_lin,
      gen_non_lin, args.embed_only, args.cuda)
  else:
    stack_model = shift_reduce_dp.ShiftReduceDP(vocab_size, args.embedding_size,
      args.hidden_size, args.num_layers, args.dropout,
      args.init_weight_range, num_features, non_lin,
      gen_non_lin, args.stack_next, args.embed_only, args.cuda)

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

    start_time = time.time()
    print('Training size {:3d}'.format(len(sentences)))

    i = 0  
    while i < len(sentences):
      # Training loop
      length = len(sentences[i])
      j = i + 1
      while (j < len(sentences) and len(sentences[j]) == length
             and (j - i) < batch_size):
        j += 1
      # dimensions [length x batch]
      sentence_data = nn_utils.get_sentence_data_batch(
          [sentences[k] for k in range(i, j)], args.cuda)
      local_batch_size = j - i
      i = j
      batch_count += 1

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

    val_loss = None
    if True:
      # Eval dev set ppl  
      val_batch_size = 1
      total_loss = 0
      total_length = 0
      total_length_more = 0
      dev_losses = []
      decode_start_time = time.time()

      stack_model.eval()
      normalize = nn.Sigmoid() 

      for val_sent in dev_sentences:
        sentence_data = nn_utils.get_sentence_data_batch([val_sent], args.cuda,
            evaluation=True)
        loss = stack_model.neg_log_likelihood(sentence_data)
        total_loss += loss.data
        dev_losses.append(loss.data[0])
        total_length += len(val_sent) - 1 
        total_length_more += len(val_sent) 

      val_loss = total_loss[0] / total_length
      val_loss_more = total_loss[0] / total_length_more
      print('-' * 89)
      print('| eval time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
          (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
      print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(val_loss_more,
          math.exp(val_loss_more)))

      print('-' * 89)


    out_name = args.working_dir + '/' + args.dev_name + '.' + str(epoch) 
    training_decode(dev_sentences, stack_model, word_vocab, out_name + '.conll',
        out_name + '.tr', use_cuda=args.cuda)
    
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

