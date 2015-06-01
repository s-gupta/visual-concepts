### From Captions to Visual Concepts and Back ### 
Code for detecting visual concepts in images.

#### Installation Instructions ####
0. Create directory, checkout code, caffe, coco-hooks
  ```shell
  git clone git@github.com:s-gupta/im2cap.git code
  git clone git@github.com:pdollar/coco.git coco
  ```
0. Make caffe and pycaffe 
  ```shell
  git clone git@github.com:s-gupta/caffe.git caffe 
  cd caffe 
  git checkout mil
  make -j 16
  make pycaffe
  cd
  ```
0. Get the COCO images, caffe imagenet models, pretrained models on COCO.
  ``` shell
  # Get the COCO images, splits, ground truth
  wget http://www.cs.berkeley.edu/~sgupta/captions/data/data.tgz && tar -xf data.tgz
  
  # Get the caffe imagenet models 
  wget http://www.cs.berkeley.edu/~sgupta/captions/data/caffe-data.tgz && tar -xf caffe-data.tgz
  
  # Get the pretrained models 
  wget http://www.cs.berkeley.edu/~sgupta/captions/data/pretrained-coco.tgz && tar -xf pretrained-coco.tgz 


#### Training, Testing the model ####
```cd code``` and execute relevant commands from the file scripts/scripts_all.py 
