# --------------------------------------------------------
# Written by Saurabh Gupta
# --------------------------------------------------------

import math, sys, json, os, h5py
import numpy as np
from pycoco.coco import COCO

class coco_voc():
  def __init__(self, image_set, devkit_path=None,  image_path=None):
    self._name = 'coco' + '_' + image_set
    self._image_set = image_set
    self._devkit_path = self._get_default_path() \
        if devkit_path is None else devkit_path
    self._data_path = os.path.join(self._devkit_path)

    self._image_path = os.path.join(self._data_path, 'images') \
        if image_path is None else image_path;
    
    self._image_ext = '.jpg'
    image_index_str, self._image_index = self._load_image_set_index()

    # Load the annotation file
    self._coco_caption_data = COCO(os.path.join(self._devkit_path, 'captions_trainval2014.json'));

    assert os.path.exists(self._devkit_path), \
        'COCO path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
        'COCO does not exist: {}'.format(self._data_path)

  def get_file_name(self, index):
    """
    Returns the file name with the folder in the beginning.
    """
    return os.path.join('{:02d}'.format(int(math.floor(int(index)/1e4))), '{}'.format(index))

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    pre_fix = '%02d' % int(math.floor(int(index)/1e4))
    image_path = os.path.join(self._image_path, self.get_file_name(index) + self._image_ext)
    assert os.path.exists(image_path), \
        'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    image_set_file = os.path.join(self._data_path, 'splits', 
                    self._image_set + '.ids')
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    imlist = [int(x) for x in image_index]
    return image_index, imlist #[:100]

  def _get_default_path(self):
    """
    Return the default path where COCO is expected to be installed.
    """
    return os.path.join('..', 'data');

  @property
  def name(self):
    return self._name

  @property
  def coco_caption_data(self):
    return self._coco_caption_data
  
  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def class_to_ind(self):
    return self._class_to_ind
  
  @property
  def image_index(self):
    return self._image_index

  @property
  def num_images(self):
    return len(self.image_index)

if __name__ == '__main__':
  d = coco_voc('train')
  from IPython import embed; embed()
