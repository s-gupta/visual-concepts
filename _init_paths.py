# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths."""
import os.path as osp
import sys
import platform

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print 'added {}'.format(path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, 'caffe', 'python')
add_path(caffe_path)

# Add this directory to PYTHONPATH
root_path = osp.join(this_dir, '.')
add_path(root_path)
