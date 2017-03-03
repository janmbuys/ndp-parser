
f = open('clusters-train-berk.txt', 'r')
dic = {}
for line in f:
  entry = line.strip().split('\t')
  dic[entry[1]] = int(entry[2])

vocab = sorted(dic, key=dic.get, reverse=True)

out = open('berk-vocab', 'w')
for word in vocab:
  out.write(word + ' ' + str(dic[word]) + '\n')

