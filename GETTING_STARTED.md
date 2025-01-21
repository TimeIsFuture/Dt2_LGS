## Getting Started with Dt2_LGS

The codes are built on the open-source toolkit [detectron2](https://github.com/facebookresearch/detectron2/tree/main).
This document provides a brief intro of the usage of builtin command-line tools.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](MODEL_ZOO.md),
  for example, `faster_rcnn_R_50_LGS_FPN_6e.yaml`.
2. We provide `demo.py` that is able to run builtin standard models. Run it with:
```
cd demo/
python demo.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_LGS_FPN_6e.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.


### Training & Evaluation in Command Line

We provide a script in "tools/{,plain_}train_net.py", that is made to train
all the configs provided in Dt2_LGS.
You may want to use it as a reference to write your own training script.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md),
then run:
```
cd tools/
./train_net.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_LGS_FPN_6e.yaml \
     --num-gpus 4 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01  \
     MODEL.WEIGHTS /path/to/pre-trained_model_file
```

The configs are made for 4-GPU training.
To train on 1 GPU, you may need to [change some parameters](https://arxiv.org/abs/1706.02677), e.g.:
```
./train_net.py \
  --config-file ../configs/COCO-Detection/faster_rcnn_R_50_LGS_FPN_6e.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025   \
  MODEL.WEIGHTS /path/to/pre-trained_model_file
```

For most models, CPU training is not supported.

To evaluate a model's performance, use
```
./train_net.py \
  --config-file ../configs/COCO-Detection/faster_rcnn_R_50_LGS_FPN_6e.yaml \
  --num-gpus 4  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `./train_net.py -h`.
