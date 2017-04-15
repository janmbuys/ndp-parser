
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


def select_features(input_features, indexes, use_cuda=False):
  if use_cuda:
    positions = Variable(torch.LongTensor(indexes)).cuda()
  else:
    positions = Variable(torch.LongTensor(indexes))
  
  return torch.index_select(input_features, 0, positions)


def batch_feature_selection(input_features, seq_length, use_cuda=False):
  indexes = []
  for i in range(seq_length-1):
    for j in range(i+1, seq_length):
      indexes.extend([i, j])
  if use_cuda:
    positions = Variable(torch.LongTensor(indexes)).cuda()
  else:
    positions = Variable(torch.LongTensor(indexes))

  selected_features = torch.index_select(input_features, 0, positions)
  return selected_features.view(-1, 2, input_features.size(2))


