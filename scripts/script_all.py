# Write splits and labels for training to be used for Caffe
python run_mil.py --task compute_targets \
  --train_dir output/v1/ --write_labels 1 --write_splits 1 \
  --train_set train --val_set valid1 \
  --vocab_file cachedir/v1/vocab_train.pkl 

# Launch the training
GLOG_logtostderr=1 caffe/build_mil_vader/tools/caffe.bin train -gpu 0 \
  -model output/v1/mil_finetune.prototxt \
  -solver output/v1/mil_finetune_solver.prototxt \
  -weights ../caffe-data/vgg_16_full_conv.caffemodel 2>&1 \
  | tee output/v1/training.log

# Testing the model
python run_mil.py --task test_model \
  --prototxt_deploy models/vgg/mil_finetune.prototxt.deploy \
  --model models/vgg/snapshot_iter_240000.caffemodel \
  --test_set valid1 --gpu 1

# Benchmarking the model
python run_mil.py --task eval_model --gpu 1 \ 
  --prototxt_deploy models/vgg/mil_finetune.prototxt.deploy \
  --model models/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file ../cachedir/v1/vocab_train.pkl 
  --test_set valid1

# Generating output txt files
python run_mil.py --task output_words --gpu 1 \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --test_set valid1 \
  --calibration_set valid1 \
  --vocab_file cachedir/v1/vocab_train.pkl 
