import os
import sg_utils as utils
import coco_voc
import shutil

# Make directories
for i in xrange(60):
  utils.mkdir_if_missing(os.path.join('..', 'data', 'images', '{:02d}'.format(i)))

# Copy files over
sets = ['train', 'val', 'test']
for set_ in sets:
  imdb = coco_voc.coco_voc(set_)
  for i in xrange(imdb.num_images):
    in_file = os.path.join('../data', set_ + '2014', \
      'COCO_{}2014_{:012d}.jpg'.format(set_, imdb.image_index[i])); 
    out_file = imdb.image_path_at(i)
    # print in_file, out_file
    shutil.copyfile(in_file, out_file)
    utils.tic_toc_print(1, ' Copying images [{}]: {:06d} / {:06d}\n'.format(set_, i, imdb.num_images));
