import sg_utils
from collections import Counter
from pycoco.coco import COCO
from nltk import pos_tag, word_tokenize

imset = 'train'
caps = COCO('../data/captions_train2014.json');
imgIds = caps.getImgIds()
imgIds.sort()
t = []
for i in xrange(len(imgIds)):
  annIds = caps.getAnnIds(imgIds[i]);
  anns = caps.loadAnns(annIds);
  tmp = [pos_tag( word_tokenize( str(a['caption']).lower())) for a in anns]
  # tmp = [[r + '-' + l for (l,r) in t] for t in tmp]
  # tmp = [sorted(t) for t in tmp]
  t.append(tmp)

# Make a vocabulary by computing counts of words over the whole dataset.
t = [t3 for t1 in t for t2 in t1 for t3 in t2]
vocab = Counter(elem for elem in t)
vocab = vocab.most_common()
sg_utils.save_variables('vocab_' + imset + '.pkl', [vocab], ['vocab'], overwrite = True);
