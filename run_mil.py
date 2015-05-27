#!/usr/bin/env python
import sg_utils as utils
import preprocess
import coco_voc
# import caffe
import argparse, pprint, sys
import numpy as np
from IPython.core.debugger import Tracer

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Compute targets for training')
  parser.add_argument('--gpu', dest='gpu_id',
            help='GPU device id to use [0]',
            default=0, type=int)
  parser.add_argument('--solver', dest='solver',
            help='solver prototxt',
            default=None, type=str)
  parser.add_argument('--iters', dest='max_iters',
            help='number of iterations to train',
            default=240000, type=int)
  parser.add_argument('--weights', dest='pretrained_model',
            help='initialize with pretrained model weights',
            default=None, type=str)
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file',
            default=None, type=str)
  parser.add_argument('--train_set', dest='train_set',
            help='dataset to train on',
            default='train', type=str)
  parser.add_argument('--val_set', dest='val_set',
            help='dataset to validate on',
            default='valid1', type=str)
  parser.add_argument('--vocab_file', dest='vocab_file',
            help='vocabulary to train for',
            default='cachedir/v1/vocab_train.pkl', type=str)
  parser.add_argument('--rand', dest='randomize',
            help='randomize (do not use a fixed seed)',
            action='store_true')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  # if not args.randomize:
  #   # fix the random seeds (numpy and caffe) for reproducibility
  #   np.random.seed(cfg.RNG_SEED)
  #   caffe.set_random_seed(cfg.RNG_SEED)

  # # set up caffe
  # caffe.set_mode_gpu()
  # caffe.set_logging_level(0)
  # if args.gpu_id is not None:
  #   caffe.set_device(args.gpu_id)
 
  if args.task == 'compute_targets':
    # Load the vocabulary
    vocab = utils.load_variables(args.vocab_file)
    
    imdb = []
    for i, imset in enumerate([args.train_set, args.val_set]):
      imdb.append(coco_voc.coco_voc(imset))
      print 'Loaded dataset `{:s}`'.format(imdb[i].name)
      
      # Compute targets for the file
      counts = preprocess.get_vocab_counts(imdb[i].image_index, \
          imdb[i].coco_caption_data, 5, vocab)
      
      if args.write_labels:

      if args.write_splits:

      # Print the command to start training

  if args.task == 'test_mdoel':
    imdb = coco_voc.coco_voc(args.test_set)
    model = load_model(directory, args.snapshot)
    test_model(imdb, model, detection_file = )

  if args.task == 'eval_model':
    benchmark(imdb, vocab, gt_label, num_references, detection_file, eval_file = None)

  if args.task == 'output_words':
    output_words(detection_file, eval_file, functional_words, threshold_metric, output_metric)
