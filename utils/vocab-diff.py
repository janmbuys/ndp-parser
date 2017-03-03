import sys

fn1, fn2 = sys.argv[1], sys.argv[2]

total1 = 0
total1_unk = 0

dic1 = {} # rnng
for line in open(fn1):
  entry = line.strip().split(' ')
  word = entry[0]
  word = word.replace('\\/', '/')
  word = word.replace('\\*', '*')
  dic1[word] = int(entry[1])
  total1 += int(entry[1])
  if word.startswith('UNK'):
    total1_unk += int(entry[1])

print(total1)
print(total1_unk)

total2 = 0
total2_unk = 0

dic2 = {} # dpdp 
for line in open(fn2):
  entry = line.strip().split(' ')
  word = entry[0]
  if word == '*root*':
    continue
  dic2[word] = int(entry[1])
  total2 += int(entry[1])
  if word.startswith('UNK'):
    total2_unk += int(entry[1])

print(total2)
print(total2_unk)

vocab1 = set(dic1.keys())
vocab2 = set(dic2.keys())

vocab_intersect = vocab1.intersection(vocab2)
vocab_diff1 = vocab1 - vocab_intersect
for word in vocab_diff1:
  print(word + ' ' + str(dic1[word]))
vocab_diff2 = vocab2 - vocab_intersect
print('')
for word in vocab_diff2:
  print(word + ' ' + str(dic2[word]))

out = open('vocab.diff', 'w')
for word in vocab_intersect:
  if dic1[word] != dic2[word]: # and word.startswith('UNK'):
    out.write(word + ' ' + str(dic1[word]) + ' ' + str(dic2[word]) + '\n')


