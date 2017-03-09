# Author: Jan Buys
# Code credit: Tensorflow seq2seq; BIST parser; pytorch master source

from collections import Counter
from pathlib import Path
import re

import torch

_SH = 0
_LA = 1
_RA = 2

class ParseForest:
  def __init__(self, sentence):
    self.roots = sentence

    for root in self.roots:
      root.children = []
      root.scores = None
      root.parent = None
      root.pred_parent_id = 0 # None
      root.pred_relation = 'rroot' # None
      root.vecs = None
      root.lstms = None

  def __len__(self):
    return len(self.roots)

  def Attach(self, parent_index, child_index):
    parent = self.roots[parent_index]
    child = self.roots[child_index]

    child.pred_parent_id = parent.id
    del self.roots[child_index]


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


# Stanford/Berkeley parser UNK processing case 5 (English specific).
# Source class: edu.berkeley.nlp.PCFGLA.SimpleLexicon
def map_unk_class(word, is_sent_start, vocab, replicate_rnng=False):
  unk_class = 'UNK'
  num_caps = 0
  has_digit = False
  has_dash = False
  has_lower = False

  if replicate_rnng:
    # Replicating RNNG bug
    for ch in word:
      has_digit = ch.isdigit()
      has_dash = ch == '-'
      if ch.isalpha():
        has_lower = ch.islower()
        if not ch.islower():
          num_caps += 1
  else:
    for ch in word:
      has_digit = has_digit or ch.isdigit()
      has_dash = has_dash or ch == '-'
      if ch.isalpha():
        has_lower = has_lower or ch.islower() or ch.istitle() 
        if not ch.islower():
          num_caps += 1

  lowered = word.lower()
  if word[0].isupper() or word[0].istitle():
    if is_sent_start and num_caps == 1:
      unk_class += '-INITC'
      if lowered in vocab:
        unk_class += '-KNOWNLC'
    else:
      unk_class += '-CAPS'
  elif not word[0].isalpha() and num_caps > 0:
    unk_class += '-CAPS'
  elif has_lower:
    unk_class += '-LC'

  if has_digit:
    unk_class += '-NUM'
  if has_dash:
    unk_class += '-DASH'

  if len(word) >= 3 and lowered[-1] == 's':
    ch2 = lowered[-2]
    if ch2 != 's' and ch2 != 'i' and ch2 != 'u':
      unk_class += '-s'
  elif len(word) >= 5 and not has_dash and not (has_digit and num_caps > 0):
    # common discriminating suffixes
    suffixes = ['ed', 'ing', 'ion', 'er', 'est', 'ly', 'ity', 'y', 'al']
    for suf in suffixes:
      if lowered.endswith(suf):
        unk_class += '-' + suf
        break

  return unk_class


def oracle(conll_sentence):
  stack = ParseForest([])
  buf = ParseForest([conll_sentence[0]])
  buffer_index = 0
  sent_length = len(conll_sentence)

  num_children = [0 for _ in conll_sentence]
  for token in conll_sentence:
    num_children[token.parent_id] += 1

  actions = []
  labels = []

  while buffer_index < sent_length or len(stack) > 1:
    action = _SH
    if buffer_index == sent_length:
      action = _RA
    elif len(stack) > 1: # allowed to ra or la
      s0 = stack.roots[-1].id 
      s1 = stack.roots[-2].id 
      b = buf.roots[0].id 
      if stack.roots[-1].parent_id == b: # candidate la
        if len(stack.roots[-1].children) == num_children[s0]:
          action = _LA 
      elif stack.roots[-1].parent_id == s1: # candidate ra
        if len(stack.roots[-1].children) == num_children[s0]:
          action = _RA 
    # excecute action
    if action == _SH:
      label = ''
      stack.roots.append(buf.roots[0]) 
      buffer_index += 1
      if buffer_index == sent_length:
        buf = ParseForest([])
      else:
        buf = ParseForest([conll_sentence[buffer_index]])
    else:  
      assert len(stack) > 0
      label = stack.roots[-1].relation
      child = stack.roots.pop()
      if action == _LA:
        buf.roots[0].children.append(child) 
      else:
        stack.roots[-1].children.append(child)
    actions.append(action)
    labels.append(label)
  return actions, labels
 

class ConllEntry:
  def __init__(self, id, form, pos, cpos, parent_id=None, relation=None):
    self.id = id
    self.form = form
    self.norm = form 
    self.cpos = cpos.upper()
    self.pos = pos.upper()
    self.parent_id = parent_id
    self.relation = relation


def isProj(sentence):
  forest = ParseForest(sentence)
  unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

  for _ in range(len(sentence)):
    for i in range(len(forest.roots) - 1):
      if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
        unassigned[forest.roots[i+1].id]-=1
        forest.Attach(i+1, i)
        break
      if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
        unassigned[forest.roots[i].id]-=1
        forest.Attach(i, i+1)
        break

  return len(forest.roots) == 1


def read_sentences_create_vocab(conll_path, conll_name, working_path, replicate_rnng=False): #TODO add argument include_singletons=False
  wordsCount = Counter()
  posCount = Counter()
  relCount = Counter()

  conll_sentences = []
  with open(conll_path + conll_name + '.conll', 'r') as conllFP:
    for sentence in read_conll(conllFP, False, replicate_rnng):
      conll_sentences.append(sentence)
      wordsCount.update([node.form for node in sentence])
      posCount.update([node.pos for node in sentence])
      relCount.update([node.relation for node in sentence])

  # For words, replace singletons with Berkeley UNK classes
  singletons = set(filter(lambda w: wordsCount[w] == 1, wordsCount.keys()))
  print(str(len(singletons)) + ' singletons')
  form_vocab = set(filter(lambda w: wordsCount[w] > 1, wordsCount.keys()))   
 
  wordsNormCount = Counter()
  for i, sentence in enumerate(conll_sentences):
    for j, node in enumerate(sentence):
      if node.form in singletons:
        conll_sentences[i][j].norm = map_unk_class(node.form, j==1, form_vocab, replicate_rnng)
    # Also add EOS to vocab
    wordsNormCount.update([node.norm for node in sentence] + ['_EOS']) 
    #wordsNormCount.update([node.norm for node in conll_sentences[i]])
 
  norm_dict = {entry[0]: i for i, entry in enumerate(wordsNormCount.most_common())}
  print('EOS id %d' % norm_dict['_EOS'])
  tensor_sentences = extract_tensor_data(conll_sentences, norm_dict)

  write_count_vocab(working_path + 'vocab', wordsNormCount)
  write_vocab(working_path + 'pos.vocab', posCount.keys())
  write_vocab(working_path + 'rel.vocab', relCount.keys())
  write_text(working_path + conll_name + '.txt', conll_sentences)

  return (conll_sentences,
          tensor_sentences,
          wordsNormCount, 
          norm_dict, 
          posCount.keys(), 
          relCount.keys())


def read_sentences_given_vocab(conll_path, conll_name, working_path, replicate_rnng=False): 
  wordsNormCount = read_vocab(working_path + 'vocab')
  form_vocab = set(filter(lambda w: not w.startswith('UNK'), 
                          wordsNormCount.keys()))

  conll_sentences = []
  with open(conll_path + conll_name + '.conll', 'r') as conllFP:
    for sentence in read_conll(conllFP, False, replicate_rnng):
      for j, node in enumerate(sentence):
        if node.form not in form_vocab: 
          sentence[j].norm = map_unk_class(node.form, j==1, form_vocab,
                                           replicate_rnng)
      conll_sentences.append(sentence)
      
  norm_dict = {entry[0]: i for i, entry in enumerate(wordsNormCount.most_common())}
  tensor_sentences = extract_tensor_data(conll_sentences, norm_dict)

  txt_filename = working_path + conll_name + '.txt'
  txt_path = Path(txt_filename)
  #if not txt_path.is_file():
  write_text(txt_filename, conll_sentences)

  return (conll_sentences,
          tensor_sentences,
          wordsNormCount, 
          norm_dict)


def read_conll(fh, proj, replicate_rnng=False):
  dropped = 0
  read = 0
  root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', 0, 'rroot')
  tokens = [root]
  for line in fh:
    tok = line.strip().split()
    if not tok:
      if len(tokens)>1:
        if (not (replicate_rnng and tokens[0].form == '#') and
            (not proj or isProj(tokens))):
          yield tokens
        else:
          print('Non-projective sentence dropped')
          dropped += 1
        read += 1
      tokens = [root]
      id = 0
    else:
      tokens.append(ConllEntry(int(tok[0]), tok[1], tok[4], tok[3], 
                               int(tok[6]) if tok[6] != '_' else -1, tok[7]))
  if len(tokens) > 1:
    yield tokens

  print('%d dropped non-projective sentences.' % dropped)
  print('%d sentences read.' % read)

# TODO make class ParseSentence, put tree and tensor and other nice things
# inside, pass through sensible global parameters

# For now just one tensor per sentence.
def extract_tensor_data(conll_sentences, vocab): #TODO add option for data.cuda()
  data = []
  eos_id = vocab['_EOS']
  for sent in conll_sentences:
    tokens = [vocab[entry.norm] for entry in sent] 
    tokens.append(eos_id)
    ids = torch.LongTensor(tokens).view(-1, 1)
    data.append(ids)
  return data


def write_conll(fn, conll_gen):
  with open(fn, 'w') as fh:
    for sentence in conll_gen:
      for entry in sentence[1:]:
        fh.write('\t'.join([str(entry.id), entry.form, '_', entry.cpos, entry.pos, '_', str(entry.pred_parent_id), entry.pred_relation, '_', '_']))
        fh.write('\n')
      fh.write('\n')


def write_text(fn, conll_gen):
  with open(fn, 'w') as fh:
    for sentence in conll_gen:
      fh.write(' '.join([entry.norm for entry in sentence[1:]]) + '\n')


def read_vocab(fn):
  dic = {}
  with open(fn, 'r') as fh:
    for line in fh:
      entry = line.strip().split(' ')
      dic[entry[0]] = int(entry[1])
  print('Read vocab of %d words.' % len(dic))
  return Counter(dic)

def write_vocab(fn, words):
  with open(fn, 'w') as fh:
    for word in words:
      fh.write(word + '\n')

def write_count_vocab(fn, counts):
  with open(fn, 'w') as fh:
    for entry in counts.most_common():
      if entry[0] == '_EOS':
        print('EOS found')
      fh.write(entry[0] + ' ' + str(entry[1]) + '\n')

