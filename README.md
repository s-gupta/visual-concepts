### From Captions to Visual Concepts and Back 
Code for detecting visual concepts in images.



#### Installation Instructions ####
0. Create directory, checkout code, caffe, coco-hooks

  ```shell
  git clone git@github.com:s-gupta/visual-concepts.git code
  git clone git@github.com:pdollar/coco.git code/coco
  cd code/coco
  git checkout 3736a0068b6e4634563b0c5847d7783dae9bd461
  cd ../../
  ```

0. Make caffe and pycaffe 

  ```shell
  git clone git@github.com:s-gupta/caffe.git code/caffe 
  cd code/caffe 
  git checkout mil
  make -j 16
  make pycaffe
  cd ../../
  ```

0. Get the COCO images, caffe imagenet models, pretrained models on COCO.

  ``` shell
  # Get the COCO splits, ground truth
  wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/data.tgz && tar -xf data.tgz
  
  # Get the COCO images
  cd code
  
  # Download and unzip the images 
  bash scripts/script_download_coco.sh
  
  # The code assumes images to be stored heirarchically. 
  # This python scripts does the required copying.
  PYTHONPATH='.' python scripts/script_download_coco.py
  
  # Get the caffe imagenet models 
  wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/caffe-data.tgz && tar -xf caffe-data.tgz
  
  # Get the pretrained models. The precomputed results in the previous tarball
  # were not complete. v2 tar ball fixes this.
  wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/trained-coco.v2.tgz && tar -xf trained-coco.v2.tgz 
  ```

#### Demo ####
Look at ``demo.ipynb`` for a demo in IPython notebook.

#### Training, Testing the model ####
``cd code`` and execute relevant commands from the file ``scripts/scripts_all.py`` 

#### Citing
If you find this codebase useful in your research, please consider citing the following paper:

    @InProceedings{Fang_2015_CVPR,
      author = {Fang, Hao and Gupta, Saurabh and Iandola, Forrest and Srivastava, Rupesh K. and Deng, Li and Dollar, Piotr and Gao, Jianfeng and He, Xiaodong and Mitchell, Margaret and Platt, John C. and Lawrence Zitnick, C. and Zweig, Geoffrey},
      title = {From Captions to Visual Concepts and Back},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2015}
    }
