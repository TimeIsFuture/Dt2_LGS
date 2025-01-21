# coding=utf-8

import torch
from torch import nn
from detectron2.layers import ShapeSpec

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from detectron2.modeling.self_attention import SelfAttention

__all__ = ["build_sanet_backbone", "SANet"]

class SANet(Backbone):
    """
    SelfAttentionNet creates the feature hierarchy built on the base backbone ResNet.
    """
    def __init__(
        self, resnet_backbone, in_features, output_features
    ):
        super().__init__()
        assert isinstance(resnet_backbone, Backbone)

        self.resnet_backbone = resnet_backbone

        self.in_features = in_features
        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = resnet_backbone.output_shape()
        in_channels = [input_shapes[f].channels for f in in_features]

        self.feature_and_sa_dict = dict()
        for stage_id, (in_chs, in_feature) in enumerate(zip(in_channels, in_features)):
            sa = SelfAttention(in_chs)
            self.add_module('%s_sa'%(in_feature), sa)
            self.feature_and_sa_dict[in_feature]=sa

        self._out_features = output_features
        self._out_feature_strides = {f: input_shapes[f].stride for f in output_features}
        self._out_feature_channels = {f: input_shapes[f].channels for f in output_features}

    def forward(self, x):
        outputs = {}
        x = self.resnet_backbone.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.resnet_backbone.stages_and_names:
            x = stage(x)
            if name in self.in_features:
                x = self.feature_and_sa_dict[name](x)
            if name in self._out_features:
                outputs[name] = x
        if self.resnet_backbone.num_classes is not None:
            x = self.resnet_backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet_backbone.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        out_shape = {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
        return out_shape


@BACKBONE_REGISTRY.register()
def build_sanet_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    resnet_backbone = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.SANET.IN_FEATURES
    output_features = cfg.MODEL.SANET.OUT_FEATURES
    extractor = SANet(
        resnet_backbone=resnet_backbone,
        in_features=in_features,
        output_features=output_features
    )
    return extractor