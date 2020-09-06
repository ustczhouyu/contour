"""Utility Scripts for CounterNet Modeling."""
import math
from typing import Dict

import cv2
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.utils.registry import Registry
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# pylint: disable=relative-beyond-top-level
from ..layers.losses import (HuberLoss, WeightedBinaryCrossEntropy,
                             WeightedMultiClassBinaryCrossEntropy)

__all__ = ["FPNBlock", "build_fpn_block", "HedDecoder", "build_hed_decoder",
           "SemSegHead", "build_sem_seg_head", 'build_contour_head',
           "CenterRegHead", "build_center_reg_head", "ContourHead",
           "build_semantic_instance_head", "SemanticInstanceHead"]


FPN_BLOCKS_REGISTRY = Registry("FPN_BLOCKS")
FPN_BLOCKS_REGISTRY.__doc__ = """ Registry for FPN BLOCKs, which make combines
 feature maps from different levels."""

HED_DECODER_REGISTRY = Registry("HED_DECODER")
HED_DECODER_REGISTRY.__doc__ = """ Registry for HED_DECODER, which generates contours
 from backbone feature maps."""

SEM_SEG_HEAD_REGISTRY = Registry("SEM_SEG_HEAD")
SEM_SEG_HEAD_REGISTRY.__doc__ = """ Registry for SEM_SEG_HEAD, which
 make semantic segmentation from FPN feature map."""

CENTER_REG_HEAD_REGISTRY = Registry("CENTER_REG_HEAD")
CENTER_REG_HEAD_REGISTRY.__doc__ = """ Registry for CENTER_REG_HEAD,
 which make center regression from FPN feature map. """

CONTOUR_HEAD_REGISTRY = Registry("CONTOUR_HEAD")
CONTOUR_HEAD_REGISTRY.__doc__ = """ Registry for CONTOUR_HEAD,
 which predict contours from FPN feature map. """

SEMANTIC_INSTANCE_HEAD_REGISTRY = Registry("SEMANTIC_INSTANCE_HEAD")
SEMANTIC_INSTANCE_HEAD_REGISTRY.__doc__ = """ Registry for
 SEMANTIC_INSTANCE_HEAD, which predicts contours and sem_seg from
 FPN feature map. """


def build_fpn(cfg, input_shape: ShapeSpec):
    """Build feature pyramid network.

    Args:
        cfg: a detectron2 CfgNode
        bottom_up(Backbone): bottom_up backbone

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    fpn = FPN(
        input_shapes=input_shape,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return fpn


def build_fpn_block(cfg, input_shape):
    """Build an FPN block from `cfg.MODEL.FPN_BLOCK.NAME`."""
    name = cfg.MODEL.FPN_BLOCK.NAME
    return FPN_BLOCKS_REGISTRY.get(name)(cfg, input_shape)


def build_hed_decoder(cfg, input_shape):
    """Build an HED decoder from `cfg.MODEL.HED_DECODER.NAME`."""
    name = cfg.MODEL.HED_DECODER.NAME
    return HED_DECODER_REGISTRY.get(name)(cfg, input_shape)


def build_sem_seg_head(cfg, input_shape):
    """Build a semantic segmentation predictior.

    Uses `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_semantic_instance_head(cfg, input_shape):
    """Build a joint semantic segmentation and contour predictior.

    Uses `cfg.MODEL.SEMANTIC_INSTANCE_HEAD.NAME`.
    """
    name = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.NAME
    return SEMANTIC_INSTANCE_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_center_reg_head(cfg, input_shape):
    """Build an center regression predictior.

    Uses `cfg.MODEL.CENTER_REG_HEAD.NAME`.
    """
    name = cfg.MODEL.CENTER_REG_HEAD.NAME
    return CENTER_REG_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_contour_head(cfg, input_shape):
    """Build an center regression predictior from `cfg.MODEL.CONTOUR_HEAD.NAME`."""
    name = cfg.MODEL.CONTOUR_HEAD.NAME
    return CONTOUR_HEAD_REGISTRY.get(name)(cfg, input_shape)


@FPN_BLOCKS_REGISTRY.register()
class FPNBlock(nn.Module):
    """Feature pyramid block.

    An FPN BLOCK similar to semantic segmentation FPN BLOCK
    described in :paper:`PanopticFPN` the predictor is removed.
    It takes FPN features as input and merges information from all
    levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features = cfg.MODEL.FPN_BLOCK.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.conv_dims = cfg.MODEL.FPN_BLOCK.CONVS_DIM
        self.common_stride = cfg.MODEL.FPN_BLOCK.COMMON_STRIDE
        norm = cfg.MODEL.FPN_BLOCK.NORM
        # fmt: on

        self.scale_blocks = []
        for in_feature in self.in_features:
            block_ops = []
            block_length = max(
                1, int(
                    np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(block_length):
                norm_module = nn.GroupNorm(
                    32, self.conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else self.conv_dims,
                    self.conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm_module,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                block_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    block_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
                    )
            self.scale_blocks.append(nn.Sequential(*block_ops))
            self.add_module(in_feature, self.scale_blocks[-1])

    # pylint: disable=arguments-differ
    def forward(self, features):
        """
        Returns:
            Merged output from all layers of FPN
        """
        for i, feature in enumerate(self.in_features):
            if i == 0:
                out = self.scale_blocks[i](features[feature])
            else:
                out = out + self.scale_blocks[i](features[feature])
        return out

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return ShapeSpec(channels=self.conv_dims, stride=self.common_stride)


# pylint: disable=too-many-instance-attributes
@HED_DECODER_REGISTRY.register()
class HedDecoder(nn.Module):
    """
    An HED (Holistic Edge Detection Network) Decoder that takes backbone inputs
     and generates contours. This is modified implementation adapted
     from HED and CASENet.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.HED_DECODER.IN_FEATURES
        n_feats = len(self.in_features)
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.loss_weight = cfg.MODEL.HED_DECODER.LOSS_WEIGHT
        norm = cfg.MODEL.HED_DECODER.NORM
        num_classes = cfg.MODEL.HED_DECODER.NUM_CLASSES
        self.num_classes = num_classes
        self.huber_active = cfg.MODEL.HED_DECODER.HUBER_ACTIVE
        self.dice_active = cfg.MODEL.HED_DECODER.DICE_ACTIVE
        self.res5_supervision = cfg.MODEL.HED_DECODER.RES5_SUPERVISION
        self.scale_blocks = []
        for in_feature in self.in_features:
            out_dims = 1
            norm_module = get_norm(norm, out_dims)
            if in_feature == self.in_features[-1]:
                out_dims = num_classes
                if self.res5_supervision:
                    norm_module = get_norm("", out_dims)

            block_stride = 2**max(1, int(np.log2(feature_strides[in_feature])))

            block_ops = []

            conv = Conv2d(
                feature_channels[in_feature],
                out_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=not norm,
                norm=norm_module,
                activation=F.relu,
            )
            weight_init.c2_msra_fill(conv)
            block_ops.append(conv)
            if block_stride != 1:
                block_ops.append(
                    nn.Upsample(scale_factor=block_stride, mode="bilinear",
                                align_corners=True)
                )
            self.scale_blocks.append(nn.Sequential(*block_ops))
            self.add_module(in_feature, self.scale_blocks[-1])

        self.predictor = nn.Conv2d(num_classes * n_feats,
                                   num_classes,
                                   kernel_size=1,
                                   groups=num_classes)
        weight_init.c2_msra_fill(self.predictor)

    # pylint: disable=arguments-differ
    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        fused, res5 = self.layers(features)
        if self.training:
            return fused, self.losses(fused, res5, targets)

        return fused, {}

    def layers(self, features):
        """Merge output from all layers of FPN.

        Args:
            features ([type]): [Output from different layers.]

        Returns:
            [type]: [Merged output from all layers.]
        """
        out = []
        for i, feat in enumerate(self.in_features):
            out.append(self.scale_blocks[i](features[feat]))
            if feat == self.in_features[-1] and self.res5_supervision:
                res5 = out[i]
        out = shared_concat(out, self.num_classes)
        out = self.predictor(out)
        if self.res5_supervision:
            return out, res5

        return out, None

    def losses(self, fused, res5, targets):
        """Compute losses."""
        if fused.size()[-2:] != targets.size()[-2:]:
            fused = F.interpolate(fused, size=targets.size()[-2:],
                                  mode="bilinear", align_corners=True)
            if self.res5_supervision:
                res5 = F.interpolate(res5, size=targets.size()[-2:],
                                     mode="bilinear", align_corners=True)
        if self.num_classes == 1:
            loss_fn = WeightedBinaryCrossEntropy(self.huber_active,
                                                 self.dice_active)
        else:
            loss_fn = WeightedMultiClassBinaryCrossEntropy(self.huber_active,
                                                           self.dice_active)
        if self.res5_supervision:
            loss_fused = loss_fn(fused, targets.squeeze())
            loss_res5 = loss_fn(res5, targets.squeeze())
            loss = {k: 0.5*loss_fused[k] + 0.5*loss_res5[k]
                    for k in loss_fused.keys()}
        else:
            loss = loss_fn(fused, targets.squeeze())
        losses = {k: v * self.loss_weight for k, v in loss.items()}
        return losses

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return ShapeSpec(channels=self.conv_dims, stride=self.common_stride)


def shared_concat(features, num_classes):
    """Perform shared concatenation."""
    n_feats = len(features)
    out_dim = num_classes * n_feats
    # pylint: disable=no-member
    out_tensor = Variable(torch.FloatTensor(features[-1].size(0),
                                            out_dim,
                                            features[-1].size(2),
                                            features[-1].size(3))).cuda()
    class_num = 0
    for i in range(0, out_dim, n_feats):
        out_tensor[:, i, :, :] = features[-1][:, class_num, :, :]
        class_num += 1
        # It needs this trick for multibatch
        for j in range(n_feats-1):
            out_tensor[:, i + j + 1, :, :] = features[j][:, 0, :, :]

    return out_tensor


@SEM_SEG_HEAD_REGISTRY.register()
class SemSegHead(nn.Module):
    """
    Semantic Segmentation predictor that takes FPN feature map as input and
     outputs semantic segmentation.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        feature_channels = input_shape.channels
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.predictor = Conv2d(feature_channels, num_classes,
                                kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    # pylint: disable=arguments-differ
    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        out = self.predictor(features)
        if self.training:
            return out, self.losses(out, targets)
        return out, {}

    def losses(self, predictions, targets):
        """Compute loss."""
        out = F.interpolate(predictions, size=targets.size()[-2:],
                            mode="bilinear", align_corners=True)
        loss = F.cross_entropy(out, targets, reduction="mean",
                               ignore_index=self.ignore_value)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@SEMANTIC_INSTANCE_HEAD_REGISTRY.register()
class SemanticInstanceHead(nn.Module):
    """
    A joint semantic segmentation and contour predictior that takes FPN
     feature map as input.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        num_classes = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.NUM_CLASSES
        feature_channels = input_shape.channels
        self.huber_active = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.HUBER_ACTIVE
        self.dice_active = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.DICE_ACTIVE
        self.loss_weight = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.LOSS_WEIGHT
        self.ignore_value = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.IGNORE_VALUE
        self.dual_loss = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.DUAL_LOSS
        if self.dual_loss:
            self.dual_loss_weight = \
                cfg.MODEL.SEMANTIC_INSTANCE_HEAD.DUAL_LOSS_WEIGHT
        self.predictor = Conv2d(feature_channels, num_classes,
                                kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    # pylint: disable=arguments-differ
    def forward(self, features, seg_targets, contour_targets):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        out = self.predictor(features)
        if self.training:
            return out, self.losses(out, seg_targets, contour_targets)

        return out, {}

    def losses(self, predictions, seg_targets, contour_targets):
        """Compute loss."""
        out = F.interpolate(predictions, size=seg_targets.size()[-2:],
                            mode="bilinear", align_corners=True)
        combined_targets = combine_seg_contour_targets(
            seg_targets, contour_targets)
        seg_loss = F.cross_entropy(out, combined_targets, reduction="mean",
                                   ignore_index=self.ignore_value)
        losses = {"loss_sem_seg": seg_loss * self.loss_weight}

        if self.dual_loss:
            contour_preds = out[:, 19:, ...]
            loss_fn = WeightedMultiClassBinaryCrossEntropy(self.huber_active,
                                                           self.dice_active)
            contour_loss = loss_fn(contour_preds, contour_targets)
            losses.update({k: v * self.loss_weight
                           for k, v in contour_loss.items()})
        return losses


@CONTOUR_HEAD_REGISTRY.register()
class ContourHead(nn.Module):
    """
    Contour predictor that takes FPN feature map as input and
     outputs Instance seperating contours.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        num_classes = cfg.MODEL.CONTOUR_HEAD.NUM_CLASSES
        feature_channels = input_shape.channels
        self.huber_active = cfg.MODEL.CONTOUR_HEAD.HUBER_ACTIVE
        self.dice_active = cfg.MODEL.CONTOUR_HEAD.DICE_ACTIVE
        self.loss_weight = cfg.MODEL.CONTOUR_HEAD.LOSS_WEIGHT
        self.predictor = Conv2d(feature_channels, num_classes,
                                kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    # pylint: disable=arguments-differ
    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        out = self.predictor(features)
        if self.training:
            return out, self.losses(out, targets)

        return out, {}

    def losses(self, predictions, targets):
        """Compute loss."""
        out = F.interpolate(predictions, size=targets.size()[-2:],
                            mode="bilinear", align_corners=True)
        loss = WeightedMultiClassBinaryCrossEntropy(self.huber_active,
                                                    self.dice_active)(out, targets)
        losses = {k: v * self.loss_weight for k, v in loss.items()}
        return losses


@CENTER_REG_HEAD_REGISTRY.register()
class CenterRegHead(nn.Module):
    """
    Center Regression predictor that takes FPN feature map as input and
     outputs center regression.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        num_classes = 2
        feature_channels = input_shape.channels
        self.loss_weight = cfg.MODEL.CENTER_REG_HEAD.LOSS_WEIGHT
        self.predictor = Conv2d(feature_channels, num_classes,
                                kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    # pylint: disable=arguments-differ
    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        out = self.predictor(features)
        if self.training:
            return out, self.losses(out, targets)
        return out, {}

    def losses(self, predictions, targets):
        """Compute loss."""
        out = F.interpolate(predictions, size=targets.size()[-2:],
                            mode="bilinear", align_corners=True)
        loss = HuberLoss()(out, targets[:, :2, :, :], targets[:, 2, :, :])
        losses = {"loss_center_reg": loss * self.loss_weight}
        return losses


class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments

    def __init__(
        self, input_shapes, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        """
        Args:
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super().__init__()

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(
            int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for i in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(i + 1)] = 2 ** (i + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {
            k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        """Get size divisibility property."""
        return self._size_divisibility

    # pylint: disable=arguments-differ
    def forward(self, _input):
        """
        Args:
            _input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        out = [_input[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](out[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            out[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2,
                                              mode="bilinear",
                                              align_corners=True)
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = _input.get(
                self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(
                    self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


# pylint: disable=too-many-locals
def get_gt(instances, image_sizes, num_classes):
    """Get ground truth for contours, offsets.

    Args:
        instances ([list]): [ground truth instances.]
        image_sizes ([list]): [list of sizes of different images.]
        num_classes ([int]): [number of contour classes. (instance classes)]

    Returns:
        [list]: [list of contour groundtruth images.]
        [list]: [list of center regression groundtruth images.]
    """
    contours, offsets = [], []
    for instance_im, size in zip(instances, image_sizes):
        contour_im = np.zeros((num_classes, size[0], size[1]),
                              dtype=np.uint8).squeeze()
        offset_im = np.zeros((3, size[0], size[1]),
                             dtype=np.float32).squeeze()
        img = np.zeros((size[0], size[1]))
        # no gt
        if not instance_im.has('gt_masks'):
            # pylint: disable=no-member
            contours.append(torch.from_numpy(contour_im))
            offsets.append(torch.from_numpy(offset_im))
            continue

        bboxes = instance_im.gt_boxes.tensor
        labels = instance_im.gt_classes
        masks = instance_im.gt_masks

        for mask, bbox, label in zip(masks, bboxes, labels):
            bit_mask = mask.cpu().numpy().astype(np.uint8)
            bbox = bbox.cpu().numpy().astype(np.uint16)
            xmin, ymin, xmax, ymax = bbox
            bit_mask_cropped = bit_mask[ymin:ymax, xmin:xmax]
            height, width = ymax - ymin, xmax - xmin
            offset_cropped = np.zeros((3, height, width),
                                      dtype=np.float32)
            g_w, g_h = np.meshgrid(np.arange(width), np.arange(height))
            x_s, y_s = np.nonzero(bit_mask_cropped)
            c_x, c_y = np.mean(x_s), np.mean(y_s)
            weight = 1000/((np.min((np.sum(bit_mask_cropped), 999))+1))
            offset_cropped[0] = (g_h - c_x)*bit_mask_cropped
            offset_cropped[1] = (g_w - c_y)*bit_mask_cropped
            offset_cropped[2] = weight*bit_mask_cropped

            # pylint: disable=no-member
            cnts, _ = cv2.findContours(
                bit_mask_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                continue
            if num_classes == 1:
                contour_im[ymin:ymax, xmin:xmax] = cv2.drawContours(
                    img[ymin:ymax, xmin:xmax], cnts, -1, 1, 2)
            else:
                contour_im[label, ymin:ymax, xmin:xmax] = cv2.drawContours(
                    img[ymin:ymax, xmin:xmax], cnts, -1, 1, 2)

            offset_im[:, ymin:ymax, xmin:xmax] += offset_cropped
        offset_im[2] += 0.01
        # pylint: disable=no-member
        contours.append(torch.from_numpy(contour_im).float())
        offsets.append(torch.from_numpy(offset_im))
    return contours, offsets


def combine_seg_contour_targets(gt_seg, gt_contours):
    """Combine semantic segmentation and contour ground truth."""
    if len(gt_contours.shape) < 4:
        gt_contours = gt_contours.unsqueeze(1)
    num_instance_classes = gt_contours.size()[1]
    combined_targets = gt_seg.clone()
    for i in range(num_instance_classes):
        gt_contours_ = gt_contours[:, i, ...].squeeze()
        idx = (gt_contours_ == 1)
        combined_targets[idx] = 19 + i
    # plt.imshow(to_rgb(combined_targets[0].cpu().numpy()))
    # plt.show()
    return combined_targets
