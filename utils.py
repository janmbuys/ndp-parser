# Author: Jan Buys
# Code credit: Tensorflow seq2seq; BIST parser

from collections import Counter
import re

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
    self.roots = list(sentence)

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

def vocab(conll_path, replicate_rnng=False): #TODO add argument include_singletons=False
  wordsCount = Counter()
  posCount = Counter()
  relCount = Counter()

  conll_sentences = []
  with open(conll_path, 'r') as conllFP:
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
    wordsNormCount.update([node.norm for node in sentence])
    #wordsNormCount.update([node.norm for node in conll_sentences[i]])

  return (wordsNormCount, 
          form_vocab,
          {w: i for i, w in enumerate(wordsNormCount.keys())},
          posCount.keys(), 
          relCount.keys())

def read_conll(fh, proj, replicate_rnng=False):
  dropped = 0
  read = 0
  root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', 0, 'rroot')
  tokens = [root]
  for line in fh:
    tok = line.strip().split()
    if not tok:
      if len(tokens)>1:
        if ((not replicate_rnng or tokens[1].form != '#') # rnng interprets # as comment
            and (not proj or isProj(tokens))):
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


def write_conll(fn, conll_gen):
  with open(fn, 'w') as fh:
    for sentence in conll_gen:
      for entry in sentence[1:]:
        fh.write('\t'.join([str(entry.id), entry.form, '_', entry.cpos, entry.pos, '_', str(entry.pred_parent_id), entry.pred_relation, '_', '_']))
        fh.write('\n')
      fh.write('\n')

def write_vocab(fn, counts):
  with open(fn, 'w') as fh:
    for entry in counts.most_common():
      fh.write(entry[0] + ' ' + str(entry[1]) + '\n')

