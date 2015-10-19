# Create directory, write splits and labels for training to be used for Caffe
mkdir output/v1

python run_mil.py --task compute_targets \
  --train_dir output/v1/ --write_labels 1 --write_splits 1 \
  --train_set train --val_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 

# Command to launch the training
GLOG_logtostderr=1 caffe/build/tools/caffe.bin train -gpu 1 \
  -model output/v1/mil_finetune.prototxt \
  -solver output/v1/mil_finetune_solver.prototxt \
  -weights ../caffe-data/vgg_16_full_conv.caffemodel 2>&1 \
  | tee output/v1/training.log

# Testing the pre-trained model
python run_mil.py --task test_model \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set valid2 --gpu 1

# Testing the pre-trained model
python run_mil.py --task test_model \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set valid1 --gpu 1

# Testing the pre-trained model
python run_mil.py --task test_model \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set test --gpu 1

# Benchmarking the pre-trained model
python run_mil.py --task eval_model --gpu 1 \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set valid1

# Generating output txt files for pre-trained model
python run_mil.py --task output_words --gpu 1 \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --test_set valid1 \
  --calibration_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 

# Generating output txt files for pre-trained model
python run_mil.py --task output_words --gpu 1 \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --test_set valid2 \
  --calibration_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 

# Generating output txt files for pre-trained model
python run_mil.py --task output_words --gpu 1 \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --test_set test \
  --calibration_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 
