# Setup Builtin Datasets

Dt2_LGS has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, Dt2_LGS expects to find datasets in the structure described below.

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

The [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
contains configs and models that use these builtin datasets.

## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

Some of the builtin tests (`dev/run_*_tests.sh`) uses a tiny version of the COCO dataset,
which you can download with `./prepare_for_tests.sh`.

## Expected dataset structure for PanopticFPN:

```
coco/
  annotations/
    panoptic_{train,val}2017.json
  panoptic_{train,val}2017/  # png annotations
  panoptic_stuff_{train,val}2017/  # generated by the script mentioned below
```

Install panopticapi by:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```
Then, run `python prepare_panoptic_fpn.py`, to extract semantic annotations from panoptic annotations.

## Expected dataset structure for LVIS instance segmentation:
```
coco/
  {train,val,test}2017/
lvis/
  lvis_v0.5_{train,val}.json
  lvis_v0.5_image_info_test.json
```

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

Run `python prepare_cocofied_lvis.py` to prepare "cocofied" LVIS annotations for evaluation of models trained on the COCO dataset.

## Expected dataset structure for cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: labelTrainIds.png are created using cityscapesescript with:
```
CITYSCAPES_DATASET=$DETECTRON2_DATASETS/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
They are not needed for instance segmentation.

## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```
