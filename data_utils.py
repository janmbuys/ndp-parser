# Author: Jan Buys
# Code credit: Tensorflow seq2seq; BIST parser; pytorch master source

from collections import Counter
from pathlib import Path
import re

import torch

_SH = 0
_LA = 1
_RA = 2
_RE = 3

_SSH = 0
_SRE = 1
_DLA = 0
_DRA = 1

_LSH = 0
_LRA = 1
_URE = 0
_ULA = 1

_EOS = 0

class ConllEntry:
  def __init__(self, id, form, pos, cpos, parent_id=None, relation=None):
    self.id = id
    self.form = form
    self.norm = form 
    self.cpos = cpos.upper()
    self.pos = pos.upper()
    self.parent_id = parent_id
    self.relation = relation


class ParseForest:
  def __init__(self, sentence):
    self.roots = sentence

    for root in self.roots:
      root.children = []
      root.scores = None
      root.parent = None
      root.pred_parent_id = 0 # None
      root.pred_relation = 'rroot' # None
      root.pred_relation_id = -1
      root.vecs = None
      root.lstms = None

  def __len__(self):
    return len(self.roots)

  def Attach(self, parent_index, child_index):
    parent = self.roots[parent_index]
    child = self.roots[child_index]

    child.pred_parent_id = parent.id
    del self.roots[child_index]


class ParseSentence:
  """Container class for single example."""
  def __init__(self, conll, tokens, relations=None):
    self.conll = conll
    self.word_tensor = torch.LongTensor(tokens).view(-1, 1)

  def __len__(self):
    return len(self.conll)

  def text_line(self):
    return ' '.join([entry.norm for entry in self.conll[1:]])

  @classmethod
  def from_vocab_conll(cls, conll, word_vocab, max_length=-1):
    tokens = [word_vocab.get_id(entry.norm) for entry in conll] + [_EOS]
    if max_length > 0 and len(tokens) > max_length:
      return cls(conll[:max_length], tokens[:max_length])
    return cls(conll, tokens)

class Vocab:
  def __init__(self, word_list, counts=None):
    self.words = word_list
    self.dic = {word: i for i, word in enumerate(word_list)}
    self.counts = counts

  def __len__(self):
    return len(self.words)

  def get_word(self, i):
    return self.words[i]

  def get_id(self, word):
    return self.dic[word]

  def form_vocab(self):
    return set(filter(lambda w: not w.startswith('UNK'), 
                      self.words))

  def write_vocab(self, fn):
    with open(fn, 'w') as fh:
      for word in self.words:
        fh.write(word + '\n')

  def write_count_vocab(self, fn, add_eos):
    assert self.counts is not None
    with open(fn, 'w') as fh:
      for i, word in enumerate(self.words):
        if i == 0 and add_eos:
          fh.write(word + ' 0\n')
        else:  
          fh.write(word + ' ' + str(self.counts[word]) + '\n')

  @classmethod
  def from_counter(cls, counter, add_eos=False):
    if add_eos:
      word_list = ['_EOS']
    else:
      word_list = []
    word_list.extend([entry[0] for entry in counter.most_common()])
    return cls(word_list, counter)

  @classmethod
  def read_vocab(cls, fn):
    with open(fn, 'r') as fh:
      word_list = []
      for line in fh:
        entry = line.strip().split(' ')
        word_list.append(entry[0])
    return cls(word_list)

  @classmethod
  def read_count_vocab(cls, fn):
    with open(fn, 'r') as fh:
      word_list = []
      dic = {}
      for line in fh:
        entry = line.strip().split(' ')
        word_list.append(entry[0])
        dic[entry[0]] = int(entry[1])
    return cls(word_list, Counter(dic))


def create_length_histogram(sentences):
  token_count = 0
  missing_token_count = 0

  sent_length = defaultdict(int)
  for sent in sentences:
    sent_length[len(sent)] += 1
    token_count += len(sent)
    missing_token_count += min(len(sent), 50)
  lengths = list(sent_length.keys())
  lengths.sort()
  print('Token count %d. length 50 count %d prop %.4f.'
        % (token_count, missing_token_count,
            missing_token_count/token_count))

  cum_count = 0
  with open(working_path + 'histogram', 'w') as fh:
    for length in lengths:
      cum_count += sent_length[length]
      fh.write((str(length) + '\t' + str(sent_length[length]) + '\t' 
                + str(cum_count) + '\n'))
  print('Created histogram')   


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


def static_oracle(conll_sentence):
  # Keep for reference
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
 

def traverse_inorder(sentence, children, i):
  """Find inorder traversal rooted at position i."""
  #print(i)
  k = 0
  order = []
  while k < len(children[i]) and children[i][k] < i:
    order.extend(traverse_inorder(sentence, children, children[i][k]))
    k += 1
  order.append(i)
  while k < len(children[i]):
    order.extend(traverse_inorder(sentence, children, children[i][k]))
    k += 1
  return order


def isProjOrder(sentence):
  children = [[] for _ in sentence]
  for i in range(1, len(sentence)):
    assert sentence[i].parent_id < len(sentence), sentence[i].parent_id
    children[sentence[i].parent_id].append(i)
  order = traverse_inorder(sentence, children, 0)
  if order == list(range(len(sentence))):
    return True, order
  else:
    return False, order


def read_conll(fh, projectify, replicate_rnng=False):
  dropped = 0
  read = 0
  root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', 0, 'rroot')
  tokens = [root]
  for line in fh:
    tok = line.strip().split()
    if not tok:
      if len(tokens)>1:
        is_proj = True
        if projectify:
          is_proj, order = isProjOrder(tokens)
          # Don't drop projective, modify to make projective
          while not is_proj:
            for i in range(1, len(order)):
              if order[i] < order[i-1]: # reattach to grandparent
                parent_id = tokens[order[i]].parent_id
                grandparent_id = tokens[parent_id].parent_id
                tokens[order[i]].parent_id = grandparent_id
                break
            is_proj, order = isProjOrder(tokens)
            if not is_proj and grandparent_id == 0:
              print("Cannot fix")
              break

        if (not (replicate_rnng and tokens[0].form == '#') and is_proj):
          yield tokens
        else:
          #print(' '.join(toke.form for toke in tokens))
          #print(' '.join(str(toke.parent_id) for toke in tokens))
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


def read_sentences_create_vocab(conll_path, conll_name, working_path,
    projectify=False, replicate_rnng=False, max_length=-1): 
    #TODO add argument include_singletons=False
  wordsCount = Counter()
  posCount = Counter()
  relCount = Counter()

  conll_sentences = []
  with open(conll_path + conll_name + '.conll', 'r') as conllFP:
    for sentence in read_conll(conllFP, projectify, replicate_rnng):
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
        conll_sentences[i][j].norm = map_unk_class(node.form, j==1, 
            form_vocab, replicate_rnng)
    wordsNormCount.update([node.norm for node in conll_sentences[i]])
                             
  word_vocab = Vocab.from_counter(wordsNormCount, add_eos=True)
  pos_vocab = Vocab.from_counter(posCount)
  rel_vocab = Vocab.from_counter(relCount)

  word_vocab.write_count_vocab(working_path + 'vocab', add_eos=True)
  pos_vocab.write_vocab(working_path + 'pos.vocab')
  rel_vocab.write_vocab(working_path + 'rel.vocab')

  parse_sentences = []
  for sent in conll_sentences:
    for j, node in enumerate(sent): 
      sent[j].relation_id = rel_vocab.get_id(node.relation) 
      sent[j].word_id = word_vocab.get_id(node.norm) 
    if len(sent) <= max_length: #TODO temp hard restriction
      parse_sentences.append(ParseSentence.from_vocab_conll(sent, word_vocab))

  write_text(working_path + conll_name + '.txt', parse_sentences)

  return (parse_sentences,
          word_vocab,
          pos_vocab,
          rel_vocab)


def read_sentences_given_vocab(conll_path, conll_name, working_path,
    projectify=False, replicate_rnng=False, max_length=-1): 
  word_vocab = Vocab.read_count_vocab(working_path + 'vocab')
  form_vocab = word_vocab.form_vocab()
  pos_vocab = Vocab.read_vocab(working_path + 'pos.vocab')
  rel_vocab = Vocab.read_vocab(working_path + 'rel.vocab')

  sentences = []
  with open(conll_path + conll_name + '.conll', 'r') as conllFP:
    for sentence in read_conll(conllFP, projectify, replicate_rnng):
      for j, node in enumerate(sentence):
        if node.form not in form_vocab: 
          sentence[j].norm = map_unk_class(node.form, j==1, form_vocab,
                                           replicate_rnng)
        sentence[j].relation_id = rel_vocab.get_id(node.relation) 
        sentence[j].word_id = word_vocab.get_id(node.norm) 
      if len(sentence) <= max_length: #TODO temp hard restriction
        sentences.append(ParseSentence.from_vocab_conll(sentence, word_vocab))

  txt_filename = working_path + conll_name + '.txt'
  txt_path = Path(txt_filename)
  if not txt_path.is_file():
    write_text(txt_filename, sentences)

  return (sentences,
          word_vocab,
          pos_vocab,
          rel_vocab)


def write_conll(fn, conll_gen):
  with open(fn, 'w') as fh:
    for sentence in conll_gen:
      for entry in sentence[1:]:
        fh.write('\t'.join([str(entry.id), entry.form, '_', entry.cpos, entry.pos, '_', str(entry.pred_parent_id), entry.pred_relation, '_', '_']))
        fh.write('\n')
      fh.write('\n')


def write_text(fn, sentences):
  with open(fn, 'w') as fh:
    for sentence in sentences:
      fh.write(sentence.text_line() + '\n') 


