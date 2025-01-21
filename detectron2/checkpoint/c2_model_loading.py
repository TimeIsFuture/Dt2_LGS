# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import re
import torch
from collections import OrderedDict

from fvcore.common.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from torch.nn import functional as F


def convert_basic_c2_names(original_keys):
    """
    Apply some basic name conversion to names in C2 weights.
    It only deals with typical backbone models.

    Args:
        original_keys (list[str]):
    Returns:
        list[str]: The same number of strings matching those in original_keys.
    """
    layer_keys = copy.deepcopy(original_keys)
    layer_keys = [
        {"pred_b": "linear_b", "pred_w": "linear_w"}.get(k, k) for k in layer_keys
    ]  # some hard-coded mappings

    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [re.sub("\\.b$", ".bias", k) for k in layer_keys]
    layer_keys = [re.sub("\\.w$", ".weight", k) for k in layer_keys]
    # Uniform both bn and gn names to "norm"
    layer_keys = [re.sub("bn\\.s$", "norm.weight", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.bias$", "norm.bias", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.rm", "norm.running_mean", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.running.mean$", "norm.running_mean", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.riv$", "norm.running_var", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.running.var$", "norm.running_var", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.gamma$", "norm.weight", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.beta$", "norm.bias", k) for k in layer_keys]
    layer_keys = [re.sub("gn\\.s$", "norm.weight", k) for k in layer_keys]
    layer_keys = [re.sub("gn\\.bias$", "norm.bias", k) for k in layer_keys]
    
    # stem
    layer_keys = [re.sub("^res\\.conv1\\.norm\\.", "conv1.norm.", k) for k in layer_keys]
    # to avoid mis-matching with "conv1" in other components (e.g. detection head)
    layer_keys = [re.sub("^conv1\\.", "stem.conv1.", k) for k in layer_keys]

    # blocks
    layer_keys = [k.replace(".branch1.", ".shortcut.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]

    # DensePose substitutions
    layer_keys = [re.sub("^body.conv.fcn", "body_conv_fcn", k) for k in layer_keys]
    layer_keys = [k.replace("AnnIndex.lowres", "ann_index_lowres") for k in layer_keys]
    layer_keys = [k.replace("Index.UV.lowres", "index_uv_lowres") for k in layer_keys]
    layer_keys = [k.replace("U.lowres", "u_lowres") for k in layer_keys]
    layer_keys = [k.replace("V.lowres", "v_lowres") for k in layer_keys]
    return layer_keys

def convert_c2_detectron_names(weights):
    """
    Map Caffe2 Detectron weight names to Detectron2 names.

    Args:
        weights (dict): name -> tensor

    Returns:
        dict: detectron2 names -> tensor
        dict: detectron2 names -> C2 names
    """
    logger = logging.getLogger(__name__)
    logger.info("Remapping C2 weights ......")
    original_keys = sorted(weights.keys())
    layer_keys = copy.deepcopy(original_keys)

    layer_keys = convert_basic_c2_names(layer_keys)

    # --------------------------------------------------------------------------
    # RPN hidden representation conv
    # --------------------------------------------------------------------------
    # FPN case
    # In the C2 model, the RPN hidden layer conv is defined for FPN level 2 and then
    # shared for all other levels, hence the appearance of "fpn2"
    layer_keys = [
        k.replace("conv.rpn.fpn2", "proposal_generator.rpn_head.conv") for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [k.replace("conv.rpn", "proposal_generator.rpn_head.conv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # RPN box transformation conv
    # --------------------------------------------------------------------------
    # FPN case (see note above about "fpn2")
    layer_keys = [
        k.replace("rpn.bbox.pred.fpn2", "proposal_generator.rpn_head.anchor_deltas")
        for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits.fpn2", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [
        k.replace("rpn.bbox.pred", "proposal_generator.rpn_head.anchor_deltas") for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]

    # --------------------------------------------------------------------------
    # Fast R-CNN box head
    # --------------------------------------------------------------------------
    layer_keys = [re.sub("^bbox\\.pred", "bbox_pred", k) for k in layer_keys]
    layer_keys = [re.sub("^cls\\.score", "cls_score", k) for k in layer_keys]
    layer_keys = [re.sub("^fc6\\.", "box_head.fc1.", k) for k in layer_keys]
    layer_keys = [re.sub("^fc7\\.", "box_head.fc2.", k) for k in layer_keys]
    # 4conv1fc head tensor names: head_conv1_w, head_conv1_gn_s
    layer_keys = [re.sub("^head\\.conv", "box_head.conv", k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # FPN lateral and output convolutions
    # --------------------------------------------------------------------------
    def fpn_map(name):
        """
        Look for keys with the following patterns:
        1) Starts with "fpn.inner."
           Example: "fpn.inner.res2.2.sum.lateral.weight"
           Meaning: These are lateral pathway convolutions
        2) Starts with "fpn.res"
           Example: "fpn.res2.2.sum.weight"
           Meaning: These are FPN output convolutions
        """
        splits = name.split(".")
        norm = ".norm" if "norm" in splits else ""
        if name.startswith("fpn.inner."):
            # splits example: ['fpn', 'inner', 'res2', '2', 'sum', 'lateral', 'weight']
            stage = int(splits[2][len("res") :])
            return "fpn_lateral{}{}.{}".format(stage, norm, splits[-1])
        elif name.startswith("fpn.res"):
            # splits example: ['fpn', 'res2', '2', 'sum', 'weight']
            stage = int(splits[1][len("res") :])
            return "fpn_output{}{}.{}".format(stage, norm, splits[-1])
        return name

    layer_keys = [fpn_map(k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # Mask R-CNN mask head
    # --------------------------------------------------------------------------
    # roi_heads.StandardROIHeads case
    layer_keys = [k.replace(".[mask].fcn", "mask_head.mask_fcn") for k in layer_keys]
    layer_keys = [re.sub("^\\.mask\\.fcn", "mask_head.mask_fcn", k) for k in layer_keys]
    layer_keys = [k.replace("mask.fcn.logits", "mask_head.predictor") for k in layer_keys]
    # roi_heads.Res5ROIHeads case
    layer_keys = [k.replace("conv5.mask", "mask_head.deconv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Keypoint R-CNN head
    # --------------------------------------------------------------------------
    # interestingly, the keypoint head convs have blob names that are simply "conv_fcnX"
    layer_keys = [k.replace("conv.fcn", "roi_heads.keypoint_head.conv_fcn") for k in layer_keys]
    layer_keys = [
        k.replace("kps.score.lowres", "roi_heads.keypoint_head.score_lowres") for k in layer_keys
    ]
    layer_keys = [k.replace("kps.score.", "roi_heads.keypoint_head.score.") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Done with replacements
    # --------------------------------------------------------------------------
    assert len(set(layer_keys)) == len(layer_keys)
    assert len(original_keys) == len(layer_keys)

    new_weights = {}
    new_keys_to_original_keys = {}
    for orig, renamed in zip(original_keys, layer_keys):
        new_keys_to_original_keys[renamed] = orig
        if renamed.startswith("bbox_pred.") or renamed.startswith("mask_head.predictor."):
            # remove the meaningless prediction weight for background class
            new_start_idx = 4 if renamed.startswith("bbox_pred.") else 1
            new_weights[renamed] = weights[orig][new_start_idx:]
            logger.info(
                "Remove prediction weight for background class in {}. The shape changes from "
                "{} to {}.".format(
                    renamed, tuple(weights[orig].shape), tuple(new_weights[renamed].shape)
                )
            )
        elif renamed.startswith("cls_score."):
            # move weights of bg class from original index 0 to last index
            logger.info(
                "Move classification weights for background class in {} from index 0 to "
                "index {}.".format(renamed, weights[orig].shape[0] - 1)
            )
            new_weights[renamed] = torch.cat([weights[orig][1:], weights[orig][:1]])
        else:
            new_weights[renamed] = weights[orig]

    return new_weights, new_keys_to_original_keys


def swin_converter(ckpt):

    new_ckpt = OrderedDict()
    new_keys_to_original_keys = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers.', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k
        
        new_ckpt['backbone.bottom_up.base_backbone.' + new_k] = new_v
        new_keys_to_original_keys['backbone.bottom_up.base_backbone.' + new_k] = k

    return new_ckpt, new_keys_to_original_keys


def convert_swin_names(model_state_dict, ckpt_state_dict):
    
    state_dict, new_keys_to_original_keys = swin_converter(ckpt_state_dict)

    logger = logging.getLogger(__name__)
    logger.info('absolute_pos_embed is  {} in ckpt_state_dict'.format(state_dict.get('absolute_pos_embed'))) 
    # reshape absolute position embedding
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model_state_dict['absolute_pos_embed'].size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning('Error in loading absolute_pos_embed, pass')
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [
        k for k in state_dict.keys()
        if 'relative_position_bias_table' in k
    ]
    
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        logger.info('{} in ckpt_state_dict shape: {}'.format(table_key, table_pretrained.shape)) 
        table_current = model_state_dict[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f'Error in loading {table_key}, pass')
        elif L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            table_pretrained_resized = F.interpolate(
                table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                size=(S2, S2),
                mode='bicubic')
            state_dict[table_key] = table_pretrained_resized.view(
                nH2, L2).permute(1, 0).contiguous()

    return state_dict, new_keys_to_original_keys


# Note the current matching is not symmetric.
# it assumes model_state_dict will have longer names.
def align_and_update_state_dicts(model_state_dict, ckpt_state_dict, c2_conversion=True):
    """
    Match names between the two state-dict, and update the values of model_state_dict in-place with
    copies of the matched tensor in ckpt_state_dict.
    If `c2_conversion==True`, `ckpt_state_dict` is assumed to be a Caffe2
    model and will be renamed at first.

    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(model_state_dict.keys())
    if c2_conversion == 'c2_conversion':
        ckpt_state_dict, original_keys = convert_c2_detectron_names(ckpt_state_dict)
    elif c2_conversion == 'swin':
        ckpt_state_dict, original_keys = convert_swin_names(model_state_dict, ckpt_state_dict)
    else:
        original_keys = {x: x for x in ckpt_state_dict.keys()}
    ckpt_keys = sorted(ckpt_state_dict.keys())

    def match(a, b):
        # Matched ckpt_key should be a complete (starts with '.') suffix.
        # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
        # but matches whatever_conv1 or mesh_head.whatever_conv1.
        return a == b or a.endswith("." + b)

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # ckpt_key string, if it matches
    match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
    # use the matched one with longest size in case of multiple matches
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_len_model = max(len(key) for key in model_keys) if model_keys else 1
    max_len_ckpt = max(len(key) for key in ckpt_keys) if ckpt_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    # matched_pairs (matched checkpoint key --> matched model key)
    matched_keys = {}
    for idx_model, idx_ckpt in enumerate(idxs.tolist()):
        if idx_ckpt == -1:
            continue
        key_model = model_keys[idx_model]
        key_ckpt = ckpt_keys[idx_ckpt]
        value_ckpt = ckpt_state_dict[key_ckpt]
        shape_in_model = model_state_dict[key_model].shape

        if shape_in_model != value_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_ckpt, value_ckpt.shape, key_model, shape_in_model
                )
            )
            logger.warning(
                "{} will not be loaded. Please double check and see if this is desired.".format(
                    key_ckpt
                )
            )
            continue

        model_state_dict[key_model] = value_ckpt.clone()
        if key_ckpt in matched_keys:  # already added to matched_keys
            logger.error(
                "Ambiguity found for {} in checkpoint!"
                "It matches at least two keys in the model ({} and {}).".format(
                    key_ckpt, key_model, matched_keys[key_ckpt]
                )
            )
            raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

        matched_keys[key_ckpt] = key_model
        logger.info(
            log_str_template.format(
                key_model,
                max_len_model,
                original_keys[key_ckpt],
                max_len_ckpt,
                tuple(shape_in_model),
            )
        )
    matched_model_keys = matched_keys.values()
    matched_ckpt_keys = matched_keys.keys()
    # print warnings about unmatched keys on both side
    unmatched_model_keys = [k for k in model_keys if k not in matched_model_keys]
    if len(unmatched_model_keys):
        logger.info(get_missing_parameters_message(unmatched_model_keys))

    unmatched_ckpt_keys = [k for k in ckpt_keys if k not in matched_ckpt_keys]
    if len(unmatched_ckpt_keys):
        logger.info(
            get_unexpected_parameters_message(original_keys[x] for x in unmatched_ckpt_keys)
        )
