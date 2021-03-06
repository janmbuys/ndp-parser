
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def to_var(ts, use_cuda, evaluation=False):
  if use_cuda:  
    return Variable(ts, volatile=evaluation).cuda()
  else:
    return Variable(ts, volatile=evaluation)

#TODO from here change where called
def to_numpy(dist):
  return dist.type(torch.FloatTensor).data.numpy()

# code: http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html#advanced-making-dynamic-decisions-and-the-bi-lstm-crf
def log_sum_exp_1d(vec):
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp(vec, dim):
    # sum over dim
    max_score, _ = torch.max(vec, dim, keepdim=True) 
    max_score_broadcast = max_score.expand(vec.size())
    return max_score.squeeze(dim) + torch.log(torch.sum(
        torch.exp(vec - max_score_broadcast), dim))

def select_features(input_features, indexes, use_cuda=False):
  positions = to_var(torch.LongTensor(indexes), use_cuda)
  return torch.index_select(input_features, 0, positions)

def batch_feature_selection(input_features, seq_length, use_cuda=False,
    rev=False, stack_next=False):
  left_indexes = [] 
  right_indexes = [] 
  if rev:
    for j in range(1, seq_length):
      for i in range(j):
        left_indexes.append(i)
        right_indexes.append(j-1 if stack_next else j)
  else:
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        left_indexes.append(i)
        right_indexes.append(j-1 if stack_next else j)

  left_positions = to_var(torch.LongTensor(left_indexes), use_cuda)
  right_positions = to_var(torch.LongTensor(right_indexes), use_cuda)

  left_selected_features = torch.index_select(input_features, 0,
      left_positions).view(-1, input_features.size(1), 1, input_features.size(2))
  right_selected_features = torch.index_select(input_features, 0, 
      right_positions).view(-1, input_features.size(1), 1, input_features.size(2))
  # dim seq_length, batch_size, 2, feat_size
  return torch.cat((left_selected_features, right_selected_features), 2)

def batch_feature_selection_more_context(input_features, seq_length, use_cuda=False,
    stack_next=False):
  leftmost_indexes = [] 
  left_indexes = [] 
  right_indexes = [] 
  for i in range(seq_length-1):
    for j in range(i+1, seq_length):
      for l in range(max(i, 1)):
        leftmost_indexes.append(l)
        left_indexes.append(i)
        right_indexes.append(j-1 if stack_next else j)

  leftmost_positions = to_var(torch.LongTensor(leftmost_indexes), use_cuda)
  left_positions = to_var(torch.LongTensor(left_indexes), use_cuda)
  right_positions = to_var(torch.LongTensor(right_indexes), use_cuda)

  leftmost_selected_features = torch.index_select(input_features, 0,
      leftmost_positions).view(-1, input_features.size(1), 1, input_features.size(2))
  left_selected_features = torch.index_select(input_features, 0,
      left_positions).view(-1, input_features.size(1), 1, input_features.size(2))
  right_selected_features = torch.index_select(input_features, 0, 
      right_positions).view(-1, input_features.size(1), 1, input_features.size(2))
  # dim seq_length, batch_size, 2, feat_size
  return torch.cat((leftmost_selected_features, left_selected_features, right_selected_features), 2)

def old_batch_feature_selection(input_features, seq_length, use_cuda=False,
    rev=False):
  indexes = [] 
  if rev:
    for j in range(1, seq_length):
      for i in range(j):
        indexes.extend([i, j])
  else:
    for i in range(seq_length-1):
      for j in range(i+1, seq_length):
        indexes.extend([i, j])

  if use_cuda:
    positions = Variable(torch.LongTensor(indexes)).cuda()
  else:
    positions = Variable(torch.LongTensor(indexes))

  selected_features = torch.index_select(input_features, 0, positions)
  return selected_features.view(-1, 2, input_features.size(1), input_features.size(2))


def extract_feature_positions(b, s0, s1=None, s2=None, more_context=False,
  stack_next=False):
  if more_context:
    assert s1 is not None
    if stack_next:
      return [s1, s0, max(0, b-1)]
    else:
      return [s1, s0, b]
  else:
    if stack_next:
      return [s0, max(0, b-1)]
    else:
      return [s0, b]


def filter_logits(logits, targets, float_var=False, use_cuda=False):
  logits_filtered = []
  targets_filtered = []
  for logit, target in zip(logits, targets):
    # this should handle the case where not direction logits 
    # should be predicted
    if logit is not None and target is not None: 
      logits_filtered.append(logit)
      assert target >= 0
      targets_filtered.append(target)

  #print(targets_filtered)
  if logits_filtered:
    if float_var:
      target_var = to_var(torch.FloatTensor(targets_filtered), use_cuda)
    else:  
      target_var = to_var(torch.LongTensor(targets_filtered), use_cuda)

    output = torch.cat(logits_filtered, 0)
    return output, target_var
  else:
    return None, None


def get_sentence_data_batch(source_list, use_cuda, evaluation=False):
  data_ts = torch.cat([source.word_tensor for source in source_list], 1)
  return to_var(data_ts, use_cuda, evaluation)


def get_sentence_oracle_data_batch(sentences, use_cuda, evaluation=False):
  data_ts = torch.cat([sent.word_tensor for sent in sentences], 1)
  data = to_var(data_ts, use_cuda, evaluation)
  
  # predictions: dim [length, batch_size]
  # features: dim [length, batch_size, 2]
  num_prediction_slots = len(sentences[0].predictions)
  num_feature_slots = len(sentences[0].features) 
  feats = []
  for i in range(num_feature_slots): 
    feat_ts = torch.cat([sent.features[i] for sent in sentences], 1)
    feat = to_var(feat_ts, use_cuda, evaluation)
    feats.append(feat)
  preds = []
  for i in range(num_prediction_slots):
    pred_ts = torch.cat([sent.predictions[i] for sent in sentences], 1)
    pred = to_var(pred_ts, use_cuda, evaluation)
    preds.append(pred)
  return data, feats, preds


def get_sentence_batch(source_list, use_cuda, evaluation=False):
  data_ts = torch.cat([source.word_tensor[:-1] for source in source_list], 1)
  target_ts = torch.cat([source.word_tensor[1:] for source in source_list], 1)
  data = to_var(data_ts, use_cuda, evaluation)
  target = to_var(target_ts.view(-1), use_cuda)
  return data, target


