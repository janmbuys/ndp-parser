# Author: Jan Buys
# Code credit: Tensorflow seq2seq; BIST parser; pytorch master source

from collections import Counter
from collections import defaultdict
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

_ESH = 0
_ERA = 1
_ERE = 2

_LSH = 0
_LRA = 1
_URE = 0
_ULA = 1

_EOS = 0

_LIN = 0
_RELU = 1
_TANH = 2
_SIG = 3


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
      root.pred_relation_ind = -1
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
    # Placeholders for oracle
    self.predictionss = None
    self.features = None

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
          fh.write(word + '\t0\n')
        else:  
          fh.write(word + '\t' + str(self.counts[word]) + '\n')

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
        entry = line.rstrip('\n').split('\t')
        word_list.append(entry[0])
    return cls(word_list)

  @classmethod
  def read_count_vocab(cls, fn):
    with open(fn, 'r') as fh:
      word_list = []
      dic = {}
      for line in fh:
        entry = line[:-1].rstrip('\n').split('\t')
        if len(entry) < 2:
          entry = line[:-1].strip().split()
        assert len(entry) >= 2, line
        word_list.append(entry[0])
        dic[entry[0]] = int(entry[1])
    return cls(word_list, Counter(dic))


def create_length_histogram(sentences, working_path):
  token_count = 0
  missing_token_count = 0

  sent_length = defaultdict(int)
  for sent in sentences:
    sent_length[len(sent)] += 1
    token_count += len(sent)
    missing_token_count += min(len(sent), 50)
  lengths = list(sent_length.keys())
  lengths.sort()
  print('Num Tokens: %d. Num <= 50: %d (%.2f percent).'
        % (token_count, missing_token_count,
            missing_token_count*100/token_count))

  cum_count = 0
  with open(working_path + 'train.histogram', 'w') as fh:
    for length in lengths:
      cum_count += sent_length[length]
      fh.write((str(length) + '\t' + str(sent_length[length]) + '\t' 
                + str(cum_count) + '\n'))
  print('Created histogram')   


def transition_to_str(action):
  if action == _SH:
    return 'SH'
  elif action == _LA:
    return 'LA'
  elif action == _RA:
    return 'RA'
  elif action == _RE:
    return 'RE'


def indicators_to_positions(index, num_indicators):
  positions = None
  if num_indicators == 1:
    if index == 0: # (0)
      positions = [0, 2, 1]
    elif index == 1: # (1)
      positions = [2, 0, 1]
  elif num_indicators == 2:
    if index == 0: # (0, 0)
      positions = [0, 2, 1, 3]
    elif index == 1: # (0, 1)
      positions = [0, 2, 3, 1]
    elif index == 2: # (1, 0)
      positions = [2, 0, 1, 3]
    elif index == 3: # (1, 1)
      positions = [2, 0, 3, 1]
  assert positions is not None, "Invalid indicator index"
  return positions


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


def swapNsubj(sentence, children, i):
  """Swap nsubj edges."""
  k = 0
  parent = sentence[i].parent_id
  while k < len(children[i]) and children[i][k] < i:
    child = children[i][k]
    if sentence[child].relation == 'nsubj':
      head_relation = sentence[i].relation
      sentence[i].parent_id = child
      sentence[i].relation = 'nsubj-of'
      sentence[child].parent_id = parent
      sentence[child].relation = head_relation
      for l in range(k):
        sibling = children[i][l]
        sentence[sibling].parent_id = child
    else:
      swapNsubj(sentence, children, child)
    k += 1
  while k < len(children[i]):
    swapNsubj(sentence, children, children[i][k])
    k += 1


def reheadSubject(sentence):
  #print('xx')
  children = [[] for _ in sentence]
  for i in range(1, len(sentence)):
    assert sentence[i].parent_id < len(sentence), sentence[i].parent_id
    children[sentence[i].parent_id].append(i)
  swapNsubj(sentence, children, 0)
 

def read_conll(fh, projectify, replicate_rnng=False, pos_only=False,
        swap_subject=False):
  dropped = 0
  non_proj = 0
  read = 0
  root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', 0, 'rroot')
  tokens = [root]
  for line in fh:
    line = line.rstrip('\n')
    if not line:
      if len(tokens)>1:
        if swap_subject:
          reheadSubject(tokens)
        is_proj = True
        if projectify:
          is_proj, order = isProjOrder(tokens)
          if not is_proj:
            non_proj += 1
          # Don't drop projective, modify to make projective
          update = True
          while (not is_proj and update):
            update = False
            for i in range(1, len(order)):
              if order[i] < order[i-1]: # reattach to grandparent
                #print(i)
                parent_id = tokens[order[i]].parent_id
                grandparent_id = tokens[parent_id].parent_id
                tokens[order[i]].parent_id = grandparent_id
                update = True
                break
            is_proj, order = isProjOrder(tokens)
            if not is_proj and grandparent_id == 0:
              #print("Cannot fix")
              break

        if (not (replicate_rnng and tokens[0].form == '#') and is_proj):
          yield tokens
        else:
          dropped += 1
        read += 1
      tokens = [root]
      id = 0
    else:
      tok = line.split('\t')
      if pos_only:
        word = tok[4]
      else:
        word = tok[1]
      tokens.append(ConllEntry(int(tok[0]), word, tok[4], tok[3], 
                               int(tok[6]) if tok[6] != '_' else -1, tok[7]))
  if len(tokens) > 1:
    yield tokens

  if dropped > 0: 
    print('%d dropped non-projective sentences.' % dropped)
  print('%d non-projective sentences. (%.2f percent) ' % 
      (non_proj, non_proj*100/read))
  print('%d sentences read.' % read)

def read_sentences_txt_given_fixed_vocab(txt_path, txt_name, working_path):
  word_vocab = Vocab.read_count_vocab(working_path + 'vocab')

  print('reading')
  sentences = []
  with open(txt_path + txt_name + '.txt', 'r') as txtFP:
    for line in txtFP:
      root = ConllEntry(0, '*root*', '_', '_')
      tokens = [root]
      for word in line.split():
        tokens.append(ConllEntry(len(tokens), word, '_', '_'))
      for j, node in enumerate(tokens):
        assert node.form in word_vocab
        tokens[j].word_id = word_vocab.get_id(node.form)   
      sentences.append(ParseSentence.from_vocab_conll(tokens, word_vocab))

  print('%d sentences read' % len(sentences))
  return (sentences, word_vocab)


def read_sentences_txt_fixed_vocab(txt_path, txt_name, working_path):
  wordsCount = Counter()

  conll_sentences = []
  with open(txt_path + txt_name + '.txt', 'r') as txtFP:
    for line in txtFP:
      root = ConllEntry(0, '*root*', '_', '_')
      tokens = [root]
      for word in line.split():
        tokens.append(ConllEntry(len(tokens), word, '_', '_'))
      wordsCount.update([node.form for node in tokens])
      conll_sentences.append(tokens)
  print('%d sentences read' % len(conll_sentences))
  word_vocab = Vocab.from_counter(wordsCount, add_eos=True)
  word_vocab.write_count_vocab(working_path + 'vocab', add_eos=True)

  parse_sentences = []
  for sent in conll_sentences:
    for j, node in enumerate(sent): 
      sent[j].word_id = word_vocab.get_id(node.norm) 
    parse_sentences.append(ParseSentence.from_vocab_conll(sent, word_vocab))
  
  return (parse_sentences, word_vocab)


def read_sentences_create_vocab(conll_path, conll_name, working_path,
    projectify=False, use_unk_classes=True, replicate_rnng=False, 
    pos_only=False, max_length=-1, swap_subject=False): 
  wordsCount = Counter()
  posCount = Counter()
  relCount = Counter()

  conll_sentences = []
  with open(conll_path + conll_name + '.conll', 'r') as conllFP:
    for sentence in read_conll(conllFP, projectify, replicate_rnng, pos_only, swap_subject):
      #if max_length <= 0 or len(sentence) <= max_length:
      conll_sentences.append(sentence)
      wordsCount.update([node.form for node in sentence])
      posCount.update([node.pos for node in sentence])
      relCount.update([node.relation for node in sentence])

  # For words, replace singletons with Berkeley UNK classes
  singletons = set(filter(lambda w: wordsCount[w] == 1, wordsCount.keys()))
  form_vocab = set(filter(lambda w: wordsCount[w] > 1, wordsCount.keys()))   

  wordsNormCount = Counter()
  for i, sentence in enumerate(conll_sentences):
    for j, node in enumerate(sentence):
      if node.form in singletons:
        if use_unk_classes:
          conll_sentences[i][j].norm = map_unk_class(node.form, j==1, 
              form_vocab, replicate_rnng)
        else:
          conll_sentences[i][j].norm = 'UNK'
    wordsNormCount.update([node.norm for node in conll_sentences[i]])
                             
  word_vocab = Vocab.from_counter(wordsNormCount, add_eos=True)
  pos_vocab = Vocab.from_counter(posCount)
  rel_vocab = Vocab.from_counter(relCount)

  print(str(len(singletons)) + ' singletons')
  print('Word vocab size %d' % len(word_vocab))
  print('POS vocab size %d' % len(pos_vocab))
  print('Relation vocab size %d' % len(rel_vocab))

  word_vocab.write_count_vocab(working_path + 'vocab', add_eos=True)
  pos_vocab.write_vocab(working_path + 'pos.vocab')
  rel_vocab.write_vocab(working_path + 'rel.vocab')

  parse_sentences = []
  for sent in conll_sentences:
    for j, node in enumerate(sent): 
      sent[j].relation_id = rel_vocab.get_id(node.relation) 
      sent[j].word_id = word_vocab.get_id(node.norm) 
    parse_sentences.append(ParseSentence.from_vocab_conll(sent, word_vocab,
        max_length))

  write_text(working_path + conll_name + '.txt', parse_sentences)
  write_conll_gold_norm(working_path + conll_name + '.conll', conll_sentences)

  return (parse_sentences,
          word_vocab,
          pos_vocab,
          rel_vocab)


def read_sentences_given_vocab(conll_path, conll_name, working_path, 
    projectify=False, use_unk_classes=True, replicate_rnng=False, 
    pos_only=False, max_length=-1, swap_subject=False): 
  word_vocab = Vocab.read_count_vocab(working_path + 'vocab')
  form_vocab = word_vocab.form_vocab()
  pos_vocab = Vocab.read_vocab(working_path + 'pos.vocab')
  rel_vocab = Vocab.read_vocab(working_path + 'rel.vocab')

  sentences = []
  conll_sentences = []
  with open(conll_path + conll_name + '.conll', 'r') as conllFP:
    for sentence in read_conll(conllFP, projectify, replicate_rnng, pos_only, swap_subject):
      #if max_length <= 0 or len(sentence) <= max_length:
      conll_sentences.append(sentence)
      for j, node in enumerate(sentence):
        if node.form not in form_vocab: 
          if use_unk_classes:
            sentence[j].norm = map_unk_class(node.form, j==1, form_vocab,
                                             replicate_rnng)
          else:
            sentence[j].norm = 'UNK'
        if node.relation in rel_vocab.dic:
          sentence[j].relation_id = rel_vocab.get_id(node.relation) 
        else: # back off to most frequent relation 
          sentence[j].relation_id = 0
        if sentence[j].norm in word_vocab.dic:
          sentence[j].word_id = word_vocab.get_id(sentence[j].norm)
        else: # back off to least frequent word
          sentence[j].word_id = len(word_vocab) - 1
          sentence[j].norm = word_vocab.get_word(sentence[j].word_id)
      sentences.append(ParseSentence.from_vocab_conll(sentence, word_vocab,
        max_length))

  write_text(working_path + conll_name + '.txt', sentences)
  write_conll_gold_norm(working_path + conll_name + '.conll', conll_sentences)

  return (sentences,
          word_vocab,
          pos_vocab,
          rel_vocab)


def read_sentences_given_fixed_vocab(conll_name, working_path, max_length=-1): 
  word_vocab = Vocab.read_count_vocab(working_path + 'vocab')
  pos_vocab = Vocab.read_vocab(working_path + 'pos.vocab')
  rel_vocab = Vocab.read_vocab(working_path + 'rel.vocab')

  sentences = []
  conll_sentences = []
  with open(working_path + conll_name + '.conll', 'r') as conllFP:
    for sentence in read_conll(conllFP, False, False, False):
      conll_sentences.append(sentence)
      for j, node in enumerate(sentence):
        if node.relation in rel_vocab.dic:
          sentence[j].relation_id = rel_vocab.get_id(node.relation) 
        else: # back off to most frequent relation 
          sentence[j].relation_id = 0
        #assert node.form in word_vocab.dic, node.form + " not in vocab"
        sentence[j].word_id = word_vocab.get_id(sentence[j].norm)
      sentences.append(ParseSentence.from_vocab_conll(sentence, word_vocab,
        max_length))

  return (sentences,
          word_vocab,
          pos_vocab,
          rel_vocab)


def write_conll_baseline(fn, conll_gen):
  with open(fn, 'w') as fh:
    for sentence in conll_gen:
      for i, entry in enumerate(sentence):
        if i > 0:
          #if entry.parent_id > entry.id:
          #  pred_parent = entry.id + 1
          #else:
          pred_parent = entry.id - 1
          fh.write('\t'.join([str(entry.id), entry.form, '_', entry.cpos,
            entry.pos, '_', str(pred_parent), '_', '_', '_']))
          fh.write('\n')
      fh.write('\n')


def write_conll_gold_norm(fn, conll_gen):
  with open(fn, 'w') as fh:
    for sentence in conll_gen:
      for entry in sentence[1:]:
        fh.write('\t'.join([str(entry.id), entry.norm, '_', entry.cpos, entry.pos, '_', str(entry.parent_id), entry.relation, '_', '_']))
        fh.write('\n')
      fh.write('\n')


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


