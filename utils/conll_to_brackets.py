import sys

def is_punct(w):
  punct = ['``', "''", '-LRB-', '-RRB-', ',', '.', ':', ';']
  return w in punct

class ConllEntry:
  def __init__(self, id, form, pos, cpos, parent_id=None, relation=None):
    self.id = id
    self.form = form
    self.norm = form 
    self.cpos = cpos.upper()
    self.pos = pos.upper()
    self.parent_id = parent_id
    self.relation = relation

def read_conll(fh):
  read = 0
  root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', -1, 'rroot')
  tokens = [root]
  for line in fh:
    line = line.rstrip('\n')
    if not line:
      if len(tokens)>1:
        yield tokens
        read += 1
      tokens = [root]
      id = 0
    else:
      tok = line.split('\t')
      tokens.append(ConllEntry(int(tok[0]), tok[1], tok[4], tok[3], 
                               int(tok[6]) if tok[6] != '_' else -1, tok[7]))
  if len(tokens) > 1:
    yield tokens
  print('%d sentences read.' % read)


if __name__ == '__main__':
  conll_fn = sys.argv[1] 
  bracket_fn = sys.argv[2] 
  bracket_strs = []

  with open(conll_fn, 'r') as conllFP:
    with open(bracket_fn, 'w') as bracket_fh:
      for sentence in read_conll(conllFP):
        children = [[] for _ in sentence]
        for token in sentence:
          if token.id > 0 and token.parent_id >= 0:
            children[token.parent_id].append(token.id)
       
        def bracket_str(i):
          tag = 'P' if is_punct(sentence[i].pos) else 'T' 
          terminal_s = '(' + tag + ' ' + sentence[i].form + ')'
          if not children[i]:
            return terminal_s
          else:
            s = '(TOP' if i == 0 else '(X'
            left_child_strs = [bracket_str(c) 
                for c in filter(lambda x: x < i, children[i])]
            if left_child_strs:
              s += ' ' + ' '.join(left_child_strs)
            if i > 0:
              s += ' ' + terminal_s
            right_child_strs = [bracket_str(c) 
                for c in filter(lambda x: x > i, children[i])]
            if right_child_strs:
              s += ' ' + ' '.join(right_child_strs)
            return s + ')'
        
        s = bracket_str(0)
        bracket_strs.append(s)
        bracket_fh.write(s + '\n') 

