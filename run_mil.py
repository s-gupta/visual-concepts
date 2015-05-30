#!/usr/bin/env python
import _init_paths
import os, sys
import sg_utils as utils
import preprocess
import coco_voc
from test_model import *
# import caffe
import argparse, pprint, sys
import numpy as np
from IPython.core.debugger import Tracer

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Compute targets for training')
  parser.add_argument('--task', dest='task',
            help='what to run', 
            default='', type=str)
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

  parser.add_argument('--test_set', dest='test_set',
            help='dataset to test on',
            default='valid2', type=str)
  parser.add_argument('--prototxt_deploy', dest='prototxt_deploy',
            help='deploy prototxt', 
            default='models/vgg/mil_finetune.prototxt.deploy', type=str)
  parser.add_argument('--model', dest='model',
            help='deploy prototxt', 
            default='models/vgg/snapshot_iter_240000.caffemodel', type=str)


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
  caffe.set_mode_gpu()
  if args.gpu_id is not None:
    caffe.set_device(args.gpu_id)
 
  # Load the vocabulary
  vocab = utils.load_variables(args.vocab_file)
  
  if args.task == 'compute_targets':
    
    imdb = []
    for i, imset in enumerate([args.train_set, args.val_set]):
      imdb.append(coco_voc.coco_voc(imset))
      print 'Loaded dataset `{:s}`'.format(imdb[i].name)
      
      # Compute targets for the file
      counts = preprocess.get_vocab_counts(imdb[i].image_index, \
          imdb[i].coco_caption_data, 5, vocab)
      
      if args.write_labels:
        a = None

      if args.write_splits:
        a = None
      # Print the command to start training

  if args.task == 'test_model':
    imdb = coco_voc.coco_voc(args.test_set)
    mean = np.array([[[ 103.939, 116.779, 123.68]]]);
    base_image_size = 565;
    model = load_model(args.prototxt_deploy, args.model, base_image_size, mean, vocab)
    out_dir = args.model + '_output'
    utils.mkdir_if_missing(out_dir)
    detection_file = os.path.join(out_dir, imdb.name + '_detections.pkl')
    
    test_model(imdb, model, detection_file = detection_file)

  if args.task == 'eval_model':
    imdb = coco_voc.coco_voc(args.test_set)
    gt_label = preprocess.get_vocab_counts(imdb.image_index, \
        imdb.coco_caption_data, 5, vocab)
    out_dir = args.model + '_output'
    detection_file = os.path.join(out_dir, imdb.name + '_detections.pkl')
    eval_file = os.path.join(out_dir, imdb.name + '_eval.pkl')
    benchmark(imdb, vocab, gt_label, 5, detection_file, eval_file = eval_file)

  if args.task == 'output_words':
    imdb = coco_voc.coco_voc(args.test_set)
    out_dir = args.model + '_output'
    detection_file = os.path.join(out_dir, imdb.name + '_detections.pkl')
    eval_file = os.path.join(out_dir, imdb.name + '_eval.pkl')
    
    out_dir = os.path.join(args.model + '_output', 'txt')
    utils.mkdir_if_missing(out_dir)
    prec_file = os.path.join(out_dir, imdb.name + '_prec.txt')
    sc_file = os.path.join(out_dir, imdb.name + '_sc.txt')
    
    output_words(imdb, detection_file, eval_file, vocab, \
      'prec', 'prec', 0.5, 5, output_file = prec_file, \
      functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are'])
    
    output_words(imdb, detection_file, eval_file, vocab, \
      'prec', 'sc', 0.5, 5, output_file = sc_file, \
      functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are'])
