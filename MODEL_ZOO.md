# Dt2_LGS Model Zoo

### COCO Object Detection

All COCO models are trained on the train-2017 set involving 118k images, and evaluated on validation-2017 set involving 5k images.
In the following table, one epoch consists of training on 118k COCO images.

#### Faster R-CNN:
<!--
(fb only) To update the table in vim:
1. Remove the old table: d}
2. Copy the below command to the place of the table
3. :.!bash

./gen_html_table.py --config 'COCO-Detection/faster*'{50,101}'*'{6e,12e,36e+MS}'*' 'COCO-Detection/faster*101*' --name R50-LGS-FPN R101-LGS-FPN X101-LGS-FPN Swin-t-LGS-FPN Swin-s-LGS-FPN --fields lr_sched box_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: faster_rcnn_R_50_LGS_FPN_6e -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_LGS_FPN_6e.yaml">ResNet-50+LGS</a></td>
<td align="center">6</td>
<td align="center">37.3</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: faster_rcnn_R_50_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_LGS_FPN_12e.yaml">ResNet-50+LGS</a></td>
<td align="center">12</td>
<td align="center">39.5</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: faster_rcnn_R_101_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_LGS_FPN_12e.yaml">ResNet-101+LGS</a></td>
<td align="center">12</td>
<td align="center">40.8</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
</tbody></table>

#### RetinaNet:
<!--
./gen_html_table.py --config 'COCO-Detection/retina*50*' 'COCO-Detection/retina*101*' --name R50-LGS-FPN R101-LGS-FPN --fields lr_sched box_AP
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: retinanet_R_50_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_LGS_FPN_12e.yaml">ResNet-50+LGS</a></td>
<td align="center">12</td>
<td align="center">38.3</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: retinanet_R_101_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_101_LGS_FPN_12e.yaml">ResNet-101+LGS</a></td>
<td align="center">12</td>
<td align="center">40.0</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
</tbody></table>


### COCO Instance Segmentation with Mask R-CNN
<!--
./gen_html_table.py --config 'COCO-InstanceSegmentation/mask*50*'{12e,36e}'*' 'COCO-InstanceSegmentation/mask*101*' --name R50-LGS-FPN R101-LGS-FPN X101-LGS-FPN Swin-t-LGS_FPN Swin-s-LGS_FPN --fields lr_sched box_AP mask_AP
-->



<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_LGS_FPN_12e.yaml">ResNet-50+LGS</a></td>
<td align="center">12</td>
<td align="center">40.3</td>
<td align="center">36.4</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: mask_rcnn_R_101_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_LGS_FPN_12e.yaml">ResNet-101+LGS</a></td>
<td align="center">12</td>
<td align="center">41.8</td>
<td align="center">37.7</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: mask_rcnn_X_101_32x8d_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_LGS_FPN_12e.yaml">ResNeXt-101+LGS</a></td>
<td align="center">12</td>
<td align="center">43.7</td>
<td align="center">38.8</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: mask_rcnn_swin-t-p4-w7_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_swin-t-p4-w7_LGS_FPN_12e.yaml">Swin-tiny+LGS</a></td>
<td align="center">12</td>
<td align="center">42.6</td>
<td align="center">39.3</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: mask_rcnn_swin-s-p4-w7_LGS_FPN_12e -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_swin-s-p4-w7_LGS_FPN_12e.yaml">Swin-small+LGS</a></td>
<td align="center">12</td>
<td align="center">45.5</td>
<td align="center">41.2</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>

###  COCO Instance Segmentation with Cascade Mask R-CNN:

<!--
./gen_html_table.py --config 'Misc/cascade*36e.yaml' --name "Cascade Mask R-CNN"  --fields box_AP mask_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: cascade_mask_rcnn_swin-s-p4-w7_LGS_36e_ms -->
 <tr><td align="left"><a href="configs/Misc/cascade_mask_rcnn_swin-s-p4-w7_LGS_36e_ms.yaml">Swin-small+LGS</a></td>
<td align="center">36+MS</td>
<td align="center">51.5</td>
<td align="center">44.7</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
</tbody></table>
</tbody></table>

"MS" means Multi-Scale recipe during training.

### PASCAL VOC Object Detection with Faster R-CNN

The Pascal VOC dataset has two prevalent parts: the VOC 2007 and VOC 2012. 
Faster R-CNN is trained on VOC 07 trainval set + VOC 12 trainval set, and tested on VOC 07 test set using 11-point interpolated AP.
For measuring the performance on the VOC 12 test set, the optimization is implemented on VOC 07 trainval set + VOC 12 trainval set + VOC 07 test set.


<!--
./gen_html_table.py --config 'PascalVOC-Detection/*' --name "R101-LGS-FPN, VOC" --fields box_mAP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">box<br/>mAP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: faster_rcnn_R_101_LGS_FPN_voc07_ms -->
 <tr><td align="left"><a href="configs/PascalVOC-Detection/faster_rcnn_R_101_LGS_FPN_voc07_ms.yaml">ResNet-101+LGS-FPN, VOC 07</a></td>
<td align="center">82.8</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
<!-- ROW: faster_rcnn_R_101_LGS_FPN_voc12_ms -->
 <tr><td align="left"><a href="configs/PascalVOC-Detection/faster_rcnn_R_101_LGS_FPN_voc12_ms.yaml">ResNet-101+LGS-FPN, VOC 12</a></td>
<td align="center">83.3</td>
<td align="center"><a href="https://pan.baidu.com/s/1oAjJMwW9aVgAr9eQZKMgeQ">model</a>[Code:8p1u]</td>
</tr>
</tbody></table>





