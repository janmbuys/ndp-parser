# Author: Jan Buys

import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

import data_utils
import nn_utils
import arc_hybrid_sup

def training_decode(val_sentences, stack_model, word_vocab, rel_vocab, 
        conll_output_fn, transition_output_fn, viterbi_decode, max_sents=-1, use_cuda=False):

  greedy_loss = 0
  length = 0
  length_more = 0
  stack_model.eval()
  decode_start_time = time.time()

  with open(conll_output_fn, 'w') as conll_fh:
    with open(transition_output_fn, 'w') as tr_fh:
      print("decoding %d sentences." % len(val_sentences))
      for i, val_sent in enumerate(val_sentences):
        if max_sents > 0 and i > max_sents:
          break
        sentence_data = nn_utils.get_sentence_data_batch([val_sent], use_cuda,
            evaluation=True)
        #gold_actions, _, _ = stack_model.oracle(val_sent.conll)

        actions, dependents, labels, greedy_word_loss = stack_model.forward(sentence_data,
                viterbi_decode) 
        #print(greedy_word_loss)
        greedy_loss += greedy_word_loss
        length += len(val_sent) - 1 
        length_more += len(val_sent) 

        action_str = ' '.join([data_utils.transition_to_str(act) 
                               for act in actions])
        tr_fh.write(action_str + '\n')
        for i, entry in enumerate(val_sent.conll):
          if i > 0:
            pred_parent = dependents[i]
            pred_label = labels[i] ##
            pred_relation = rel_vocab.get_word(labels[i])
            conll_fh.write('\t'.join([str(entry.id), entry.form, '_', 
              entry.cpos, entry.pos, 
              '_', str(pred_parent), pred_relation, '_', '_']) + '\n')
        conll_fh.write('\n')
  print('-' * 89)
  print('decode time {:2.2f}s'.format(time.time() - decode_start_time))

  if stack_model.generative:
    val_loss = greedy_loss / length # [0]
    val_loss_more = greedy_loss / length_more

    print('| greedy valid loss {:5.2f} | valid ppl {:8.2f} '.format(
         val_loss, math.exp(val_loss)))
    print('       | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
         val_loss_more, math.exp(val_loss_more)))
  print('-' * 89)


def sentence_generate(val_sentences, stack_model, word_vocab, rel_vocab, 
        conll_output_fn, viterbi_decode, max_sents=-1, use_cuda=False):

  greedy_loss = 0
  length = 0
  length_more = 0
  stack_model.eval()
  decode_start_time = time.time()

  with open(conll_output_fn, 'w') as conll_fh:
    print("generating %d sentences." % len(val_sentences))
    for i, val_sent in enumerate(val_sentences):
      if max_sents > 0 and i > max_sents:
        break
      sentence_data = nn_utils.get_sentence_data_batch([val_sent], use_cuda,
          evaluation=True)
      gold_actions, _, _ = stack_model.oracle(val_sent.conll)

      actions, dependents, labels, word_ids, greedy_word_loss = stack_model.generate(sentence_data,
              False) #, gold_actions) #TODO parameterize actions given
      greedy_loss += greedy_word_loss
      length += len(word_ids) - 1 
      length_more += len(word_ids) 
      #TODO store words
      sentence = [word_vocab.get_word(word) for word in word_ids]
      sentence_str = ' '.join([word_vocab.get_word(word) 
                               for word in word_ids[1:]])
      print(sentence_str)

      #action_str = ' '.join([data_utils.transition_to_str(act) 
      #                       for act in actions])
      #tr_fh.write(action_str + '\n')
      for i, word_str in enumerate(sentence):
        if i > 0:
          pred_parent = dependents[i]
          pred_label = labels[i] ##
          pred_relation = rel_vocab.get_word(labels[i])
          conll_fh.write('\t'.join([word_str, word_str, '_', 
            '_', '_', '_', str(pred_parent), pred_relation, '_', '_']) + '\n')
      conll_fh.write('\n')
  print('-' * 89)
  print('decode time {:2.2f}s'.format(time.time() - decode_start_time))

  val_loss = greedy_loss / length # [0]
  val_loss_more = greedy_loss / length_more

  print('| greedy valid loss {:5.2f} | valid ppl {:8.2f} '.format(
       val_loss, math.exp(val_loss)))
  print('       | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
       val_loss_more, math.exp(val_loss_more)))
  print('-' * 89)


def training_score(args, stack_model, val_sentences):
  total_loss = 0
  total_length = 0
  total_length_more = 0

  stack_model.eval()
  viterbi_score = False
  
  print('Scoring val sentences')
  for val_sent in val_sentences:
    sentence_data = nn_utils.get_sentence_data_batch([val_sent], args.cuda,
        evaluation=True)
    if viterbi_score:
      loss = stack_model.viterbi_neg_log_likelihood(sentence_data)
      total_loss += loss
    else:
      loss = stack_model.neg_log_likelihood(sentence_data) # inside_score
      #print(loss)
      total_loss += loss #.data[0]
    total_length += len(val_sent) - 1 
    total_length_more += len(val_sent) 
  print("Total loss %f" % float(total_loss))

  return total_loss, total_length, total_length_more


def generate(args, val_sentences, word_vocab, rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  model_path = args.working_dir + '/' + args.save_model
  non_lin = args.non_lin #TODO better interface
  gen_non_lin = args.gen_non_lin

  assert model_path != ''
  print('Loading models')
  model_fn = model_path + '.pt'
  with open(model_fn, 'rb') as f:
    stack_model = torch.load(f)
 
  if args.cuda:
    stack_model.cuda()

  print('Done loading models')

  #TODO write generated samples to file
  out_name = args.working_dir + '/' + args.dev_name 
  sentence_generate(val_sentences, stack_model, word_vocab, rel_vocab, out_name
      + '.generate.conll', args.viterbi_decode, use_cuda=args.cuda)


def decode(args, val_sentences, word_vocab, rel_vocab, score=False):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  model_path = args.working_dir + '/' + args.save_model
  non_lin = args.non_lin #TODO better interface
  gen_non_lin = args.gen_non_lin

  assert model_path != ''
  print('Loading models')
  model_fn = model_path + '.pt'
  with open(model_fn, 'rb') as f:
    stack_model = torch.load(f)
 
  if args.cuda:
    stack_model.cuda()

  print('Done loading models')

  if score:
    decode_start_time = time.time()
    total_loss, total_length, total_length_more = training_score(args,
        stack_model, val_sentences)

    val_loss = total_loss / total_length
    val_loss_more = total_loss / total_length_more

    print('-' * 89)
    print('| decoding time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} '.format(
         (time.time() - decode_start_time), val_loss, math.exp(val_loss)))
    print('                     | valid loss more {:5.2f} | valid ppl {:8.2f}'.format(
         val_loss_more, math.exp(val_loss_more)))
    print('-' * 89)
  else: 
    out_name = args.working_dir + '/' + args.dev_name 
    training_decode(val_sentences, stack_model, word_vocab, rel_vocab, out_name + '.output.conll',
        out_name + '.output.tr', args.viterbi_decode, use_cuda=args.cuda)


def train(args, sentences, dev_sentences, word_vocab, rel_vocab):
  vocab_size = len(word_vocab)
  num_relations = len(rel_vocab)
  model_path = args.working_dir + '/' + args.save_model
  lr = args.lr
  non_lin = args.non_lin #TODO better interface
  gen_non_lin = args.gen_non_lin

  # Build the model
  assert args.arc_hybrid or args.arc_eager
  if args.arc_hybrid:
    stack_model = arc_hybrid_sup.ArcHybridSup(vocab_size, num_relations,
        args.embedding_size, args.hidden_size, args.num_layers, args.dropout,
        args.init_weight_range, args.bidirectional, non_lin, gen_non_lin, 
        args.generative, args.stack_next, args.cuda)
  elif args.arc_eager:
      assert False, "Not yet implemented."
      #TODO 
      # args.late_reduce_oracle)

  if args.cuda:
    stack_model.cuda()

  params = list(stack_model.parameters()) 

  if args.adam:
    optimizer = optim.Adam(params, lr=lr)
  else:
    optimizer = optim.SGD(params, lr=lr)

  print('Training size {:3d}'.format(len(sentences)))
  # run the oracle
  for sentence in sentences:
    # (transition_action, [direction], word_gen, relation_label) tuples
    actions, sentence.predictions, sentence.features = stack_model.oracle(sentence.conll)

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
    batch_count = 0

    start_time = time.time()
    i = 0  
    while i < len(sentences):
      # Training loop
      length = len(sentences[i])
      j = i + 1
      while (j < len(sentences) and len(sentences[j]) == length
             and (j - i) < args.batch_size):
        j += 1
      local_batch_size = j - i
      sentence_data, sentence_feats, sentence_preds = nn_utils.get_sentence_oracle_data_batch(
          [sentences[k] for k in range(i, j)], args.cuda)

      i = j
      batch_count += 1

      stack_model.train()
      stack_model.zero_grad() 
      loss = stack_model.joint_neg_log_likelihood(sentence_data,
          sentence_feats, sentence_preds)

      if loss is not None:
        loss.backward()
        if args.grad_clip > 0:
          nn.utils.clip_grad_norm(params, args.grad_clip)
        optimizer.step() 
        total_loss += loss.data
        global_loss += loss.data

      batch_tokens = (sentence_data.size()[0] - 2)*local_batch_size
      total_num_tokens += batch_tokens
      global_num_tokens += batch_tokens

      if batch_count % args.logging_interval == 0 and i > 0:
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

    # Decoding
    out_name = args.working_dir + '/' + args.dev_name + '.' + str(epoch)
    training_decode(dev_sentences, stack_model, word_vocab, rel_vocab,
        out_name + '.conll', out_name + '.tr', args.viterbi_decode, use_cuda=args.cuda)

    if args.generative:
      # score the model 
      decode_start_time = time.time()
      total_loss, total_length, total_length_more = training_score(args,
          stack_model, dev_sentences)

      val_loss = total_loss / total_length
      val_loss_more = total_loss / total_length_more
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
    if args.generative and prev_val_loss and val_loss > prev_val_loss:
      patience_count += 1
    else:
      patience_count = 0
      if args.generative:
        prev_val_loss = val_loss
      # Save the model.
      if args.save_model != '':
        model_fn = args.working_dir + '/' + args.save_model + '.pt'
        with open(model_fn, 'wb') as f:
          torch.save(stack_model, f)

    if args.store_all_iterations:
      model_fn = (args.working_dir + '/' + args.save_model + '.' + str(epoch) 
                  + '.pt')
      with open(model_fn, 'wb') as f:
        torch.save(stack_model, f)

    if args.patience > 0 and patience_count >= args.patience:
      break

