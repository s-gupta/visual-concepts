import sg_utils
import numpy as np
from collections import Counter
from pycoco.coco import COCO
from nltk import pos_tag, word_tokenize, wordpunct_tokenize

imset = 'train'
caps = COCO('../data/captions_train2014.json');
imgIds = caps.getImgIds()
imgIds.sort()
t = []

# punctuations to be removed from the sentences
punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
  ".", "?", "!", ",", ":", "-", "--", "...", ";"] 

mapping = {'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN', 'NN': 'NN', \
  'VB': 'VB', 'VBD': 'VB', 'VBN': 'VB', 'VBZ': 'VB', 'VBP': 'VB', 'VBP': 'VB', 'VBG': 'VB', \
  'JJR': 'JJ', 'JJS': 'JJ', 'JJ': 'JJ', \
  'DT': 'DT', \
  'PRP': 'PRP', 'PRP$': 'PRP', \
  'IN': 'IN'};

for i in xrange(2000): #len(imgIds)):
  annIds = caps.getAnnIds(imgIds[i]);
  anns = caps.loadAnns(annIds);
  tmp = [pos_tag( wordpunct_tokenize( str(a['caption']).lower())) for a in anns]
  # tmp = [[r + '-' + l for (l,r) in t] for t in tmp]
  # tmp = [sorted(t) for t in tmp]
  t.append(tmp)

# Make a vocabulary by computing counts of words over the whole dataset.
t = [t3 for t1 in t for t2 in t1 for t3 in t2]
t = [(l, 'other') if mapping.get(r) is None else (l, mapping[r]) for (l,r) in t]
vocab = Counter(elem for elem in t)
vocab = vocab.most_common()
# sg_utils.save_variables('vocab_' + imset + '.pkl', [vocab], ['vocab'], overwrite = True);

# Merge things that are in the same or similar pos
word = [l for ((l,r),c) in vocab];
pos = [r for ((l,r),c) in vocab];
count = [c for ((l,r),c) in vocab];

poss = [];
counts = [];
words = sorted(set(word))
for j in xrange(len(words)):
  indexes = [i for i,x in enumerate(word) if x == words[j]]
  pos_tmp = [pos[i] for i in indexes]
  count_tmp = [count[i] for i in indexes]
  ind = np.argmax(count_tmp)
  poss.append(pos_tmp[ind])
  counts.append(sum(count_tmp))

ind = np.argsort(counts)
ind = ind[::-1]
words = [words[i] for i in ind]
poss = [poss[i] for i in ind]
counts = [counts[i] for i in ind]

# Remove punctuations
non_punct = [i for (i,x) in enumerate(words) if x not in punctuations]
words = [words[i] for i in non_punct]
counts = [counts[i] for i in non_punct]
poss = [poss[i] for i in non_punct]

sg_utils.save_variables('vocab_' + imset + '.pkl', \
  [words, poss, counts], \
  ['words', 'poss', 'counts'], \
  overwrite = True);
