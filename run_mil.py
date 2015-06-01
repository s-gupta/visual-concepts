#!/usr/bin/env python
import h5py, math
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
  parser = argparse.ArgumentParser(description='Script for training and testing word detection models.')

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
            default='valid1', type=str)
  parser.add_argument('--val_set', dest='val_set',
            help='dataset to validate on',
            default='valid1', type=str)
  parser.add_argument('--train_dir', dest='train_dir',
            help='directory to train the models from',
            default=None, type=str)
  parser.add_argument('--write_labels', dest='write_labels',
            default=False, type=bool)
  parser.add_argument('--write_splits', dest='write_splits',
            default=False, type=bool)

  parser.add_argument('--calibration_set', dest='calibration_set',
            help='dataset to use eval file from',
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
    output_dir = args.train_dir
    sets = ['train', 'val']
    for i, imset in enumerate([args.train_set, args.val_set]):
      imdb.append(coco_voc.coco_voc(imset))
      print 'Loaded dataset {:s}'.format(imdb[i].name)
      
      # Compute targets for the file
      counts = preprocess.get_vocab_counts(imdb[i].image_index, \
          imdb[i].coco_caption_data, 5, vocab)
      
      if args.write_labels:
        label_file = os.path.join(output_dir, 'labels_' + sets[i] + '.h5') 
        print 'Writing labels to {}'.format(label_file)
        with h5py.File(label_file, 'w') as f:
          for j in xrange(imdb[i].num_images):
            ind = imdb[i].image_index[j]
            ind_str = '{:02d}/{:d}'.format(int(math.floor(ind)/1e4), ind)
            l = f.create_dataset('/labels-{}'.format(ind_str), (1, 1, counts.shape[1], 1), dtype = 'f')
            c = counts[j,:].copy(); c = c > 0; c = c.astype(np.float32); c = c.reshape((1, 1, c.size, 1))
            l[...] = c
            utils.tic_toc_print(1, 'write labels {:6d} / {:6d}'.format(j, imdb[i].num_images)) 

      if args.write_splits:
        split_file = os.path.join(output_dir, sets[i] + '.ids') 
        print 'Writing labels to {}'.format(split_file)
        with open(split_file, 'wt') as f:
          for j in xrange(imdb[i].num_images):
            ind = imdb[i].image_index[j]
            ind_str = '{:02d}/{:d}'.format(int(math.floor(ind)/1e4), ind)
            f.write('{}\n'.format(ind_str))

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
    out_dir = args.model + '_output'
    
    imdb = coco_voc.coco_voc(args.test_set)
    detection_file = os.path.join(out_dir, imdb.name + '_detections.pkl')
    
    imdb_cal = coco_voc.coco_voc(args.calibration_set)
    eval_file = os.path.join(out_dir, imdb_cal.name + '_eval.pkl')
    
    out_dir = os.path.join(args.model + '_output', 'txt')
    utils.mkdir_if_missing(out_dir)
    prec_file = os.path.join(out_dir, imdb.name + '_prec.txt')
    sc_file = os.path.join(out_dir, imdb.name + '_sc.txt')
    
    output_words(imdb, detection_file, eval_file, vocab, \
      'prec', 'prec', 0.5, 3, output_file = prec_file, \
      functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are'])
    
    output_words(imdb, detection_file, eval_file, vocab, \
      'prec', 'sc', 0.5, 3, output_file = sc_file, \
      functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are'])
