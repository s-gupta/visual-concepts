## MATLAB vocabulary
if job_name == 'vocab':
  import csv
  import sg_utils
  matlab_vocab = 'vocabs/vocab_words.txt';
  words = []; poss = []; counts = [];
  with open(matlab_vocab, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      words.append(row[0].strip()) 
      poss.append(row[1].strip()) 
      counts.append(int(row[2].strip()))
  sg_utils.save_variables('vocabs/vocab_train.pkl', [words, poss, counts], \
    ['words', 'poss', 'counts'], overwrite = True) 

## Load the detections
# Code to re-evaluate matlab output, to check sainty of python code
# if job_name == 'eval_det':
#   import sg_utils
#   import test_model
#   import cap_eval_utils
#   vocab = sg_utils.load_variables('cachedir/v1/vocab_train.pkl')
#   dt = sg_utils.scio.loadmat('cachedir/v1/gt_labels_val.all.mat'); labels = dt['labels'];
#   details = test_model.benchmark(None, vocab, labels, 5., 'cachedir/v1/mil_prob_val.all.mat')
