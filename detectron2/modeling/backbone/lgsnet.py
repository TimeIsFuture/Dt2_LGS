# coding=utf-8

import torch
from torch import nn
from detectron2.layers import ShapeSpec

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from .swin import build_swin_backbone
from detectron2.modeling.lgs import LocalGlobalSemantics

__all__ = ["build_lgsnet_backbone", "LGSNet"]

class LGSNet(Backbone):
    """
    LGSNet creates the feature hierarchy built on the base backbone ResNet or Swin-Transformer.
    """
    def __init__(
        self, backbone_name, resnet_backbone, in_features, output_features, K=7, groups=8, heads=64
    ):
        super().__init__()
        assert isinstance(resnet_backbone, Backbone)

        self.resnet_backbone = resnet_backbone

        self.in_features = in_features
        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = resnet_backbone.output_shape()
        in_channels = [input_shapes[f].channels for f in in_features]
        name_forward_dict = {'resnet': self.resnet_forward, 'swin':self.swin_forward}
        self.backbone_forward = name_forward_dict[backbone_name]

        self.feature_and_lgs_dict = dict()
        for stage_id, (in_chs, in_feature) in enumerate(zip(in_channels, in_features)):
            lgs = LocalGlobalSemantics(in_chs, in_chs, K=K, groups=groups, heads=heads)
            self.add_module('%s_lgs'%(in_feature), lgs)
            self.feature_and_lgs_dict[in_feature] = lgs

        self._out_features = output_features
        self._out_feature_strides = {f: input_shapes[f].stride for f in output_features}
        self._out_feature_channels = {f: input_shapes[f].channels for f in output_features}

    def resnet_forward(self, x):
        outputs = {}
        x = self.resnet_backbone.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.resnet_backbone.stages_and_names:
            x = stage(x)
            if name in self.in_features:
                x = self.feature_and_lgs_dict[name](x)
            if name in self._out_features:
                outputs[name] = x
        if self.resnet_backbone.num_classes is not None:
            x = self.resnet_backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet_backbone.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def swin_forward(self, x):
        outputs = {}
        x, hw_shape = self.resnet_backbone.patch_embed(x)
        if "patch_embed" in self._out_features:
            outputs["patch_embed"] = x

        if self.resnet_backbone.use_abs_pos_embed:
            x = x + self.resnet_backbone.absolute_pos_embed
        x = self.resnet_backbone.drop_after_pos(x)
        outputs = {}
        for stage, i in self.resnet_backbone.stages_and_indices:
            for block in stage.blocks:
                x = block(x, hw_shape)
            
            name = f'swin{i}'
            num_features = self.resnet_backbone.num_features[i]
            if name in self.in_features[i]:
                out = x.view(-1, *hw_shape, num_features).permute(0, 3, 1, 2).contiguous()
                out = self.feature_and_lgs_dict[name](out)
                x = out.flatten(2).permute(0, 2, 1).contiguous()
                
            if name in self._out_features:
                norm_layer = getattr(self.resnet_backbone, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape, num_features).permute(0, 3, 1, 2).contiguous()
                outputs[name] = out

            if stage.downsample:
                x, hw_shape = stage.downsample(x, hw_shape)
        return outputs

    def forward(self, x):
        outputs = self.backbone_forward(x)
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
def build_lgsnet_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    backbone_name = cfg.MODEL.LGSNET.BACKBONE_NAME
    name_backbone_dict = {'resnet': build_resnet_backbone, 'swin': build_swin_backbone}
    build_backbone = name_backbone_dict[backbone_name]
    resnet_backbone = build_backbone(cfg, input_shape)
    in_features = cfg.MODEL.LGSNET.IN_FEATURES
    output_features = cfg.MODEL.LGSNET.OUT_FEATURES
    K = cfg.MODEL.LGSNET.K
    groups = cfg.MODEL.LGSNET.GROUPS
    heads = cfg.MODEL.LGSNET.HEADS
    extractor = LGSNet(
        backbone_name=backbone_name,
        resnet_backbone=resnet_backbone,
        in_features=in_features,
        output_features=output_features,
        K=K,
        groups=groups,
        heads=heads
    )
    return extractor