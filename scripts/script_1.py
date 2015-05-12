import numpy as np
from pycoco.coco import COCO
import preprocess
import sg_utils
import cap_eval_utils

imset = 'train'
coco_caps = COCO('../data/captions_train2014.json');

# mapping to output final statistics
mapping = {'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN', 'NN': 'NN', \
  'VB': 'VB', 'VBD': 'VB', 'VBN': 'VB', 'VBZ': 'VB', 'VBP': 'VB', 'VBP': 'VB', 'VBG': 'VB', \
  'JJR': 'JJ', 'JJS': 'JJ', 'JJ': 'JJ', 'DT': 'DT', 'PRP': 'PRP', 'PRP$': 'PRP', 'IN': 'IN'};
    
# punctuations to be removed from the sentences
punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
  ".", "?", "!", ",", ":", "-", "--", "...", ";"] 

vocab = preprocess.get_vocab(imset, coco_caps, punctuations, mapping);

sg_utils.save_variables('vocab_' + imset + '.pkl', \
  [vocab[x] for x in vocab.keys()], \
  vocab.keys(), \
  overwrite = True);


##
N_WORDS = 1000;
vocab = preprocess.get_vocab_top_k(vocab, N_WORDS)
image_ids = coco_caps.getImgIds()
counts = preprocess.get_vocab_counts(image_ids, coco_caps, 5, vocab)
P = np.zeros((N_WORDS, 1), dtype = np.float); 
R = np.zeros((N_WORDS, 1), dtype = np.float); 
for i, w in enumerate(vv['words']): 
  P[i], R[i] = cap_eval_utils.human_agreement(counts[:,i], 5)
  print w, P[i], R[i]

for pos in list(set(vocab['poss'])):
  ind = [i for i,x in enumerate(vocab['poss']) if pos == x]
  print "{:5s} [{:3d}] : {:.2f} {:.2f} ".format(pos, len(ind), 100*np.mean(P[ind]), 100*np.mean(R[ind]))
