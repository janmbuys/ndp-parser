
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return
    for p in parameters:
        p.grad.data.mul_(clip_coef)

#TODO from here change where called
def to_numpy(dist):
  return dist.type(torch.FloatTensor).data.numpy()

# code: http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html#advanced-making-dynamic-decisions-and-the-bi-lstm-crf
def log_sum_exp(vec):
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_2d(vec):
    max_score, _ = torch.max(vec, 1) 
    #max_score = vec[0, np.argmax(vec, 1)]
    max_score_broadcast = max_score.view(-1, 1).expand(vec.size())
    return max_score + torch.log(torch.sum(
        torch.exp(vec - max_score_broadcast), 1))

def select_features(input_features, indexes, use_cuda=False):
  if use_cuda:
    positions = Variable(torch.LongTensor(indexes)).cuda()
  else:
    positions = Variable(torch.LongTensor(indexes))
  
  return torch.index_select(input_features, 0, positions)


def batch_feature_selection(input_features, seq_length, use_cuda=False,
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

  #TODO I now want to select same positions, but for whole batch, and have
  # extract (first) batch dimension as output
  selected_features = torch.index_select(input_features, 0, positions)
  return selected_features.view(-1, 2, input_features.size(2))


def extract_feature_positions(b, s0, s1=None, s2=None, more_context=False):
  # Ensure feature extraction is consistent
  if more_context and s1 is not None and s2 is not None:
    return [s2, s1, s0, b]
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
    if use_cuda:
      if float_var:
        target_var = Variable(torch.FloatTensor(targets_filtered)).cuda()
      else:  
        target_var = Variable(torch.LongTensor(targets_filtered)).cuda()
    else:
      if float_var:
        target_var = Variable(torch.FloatTensor(targets_filtered))
      else:
        target_var = Variable(torch.LongTensor(targets_filtered))

    output = torch.cat(logits_filtered, 0)
    return output, target_var
  else:
    return None, None


def get_sentence_data_batch(source_list, use_cuda, evaluation=False):
  data_ts = torch.cat([source.word_tensor for source in source_list], 1)
  if use_cuda:
    data = Variable(data_ts, volatile=evaluation).cuda()
  else:
    data = Variable(data_ts, volatile=evaluation)
  return data


def get_sentence_batch(source_list, use_cuda, evaluation=False):
  data_ts = torch.cat([source.word_tensor[:-1] for source in source_list], 1)
  target_ts = torch.cat([source.word_tensor[1:] for source in source_list], 1)
  if use_cuda:
    data = Variable(data_ts, volatile=evaluation).cuda()
    target = Variable(target_ts.view(-1)).cuda()
  else:
    data = Variable(data_ts, volatile=evaluation)
    target = Variable(target_ts.view(-1))
  return data, target



