from typing import Dict
import math
import copy
import fvcore.nn.weight_init as weight_init
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, get_norm
from detectron2.structures import ImageList, polygons_to_bitmask
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling import Backbone

from ..layers.losses import HuberLoss, WeightedBinaryCrossEntropy, \
    WeightedMultiClassBinaryCrossEntropy
from .visualization import rgb_from_gt_contours, rgb_from_pred_contours
import matplotlib.pyplot as plt
from .postprocessing import to_rgb


__all__ = ["FPNBLOCK", "build_fpn_BLOCK", "HEDHead", "build_hed_head",
           "SemSegHead", "build_sem_seg_head", 'build_contour_head',
           "CenterRegHead", "build_center_reg_head", "ContourHead",
           "build_semantic_instance_head", "SemanticInstanceHead"]


FPN_BLOCKS_REGISTRY = Registry("FPN_BLOCKS")
FPN_BLOCKS_REGISTRY.__doc__ = """ Registry for FPN BLOCKs, which make combines
 feature maps from different levels."""

HED_HEAD_REGISTRY = Registry("HED_HEAD")
HED_HEAD_REGISTRY.__doc__ = """ Registry for HED_HEAD, which generates contours
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
    """
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
    """
    Build an FPN block from `cfg.MODEL.FPN_BLOCK.NAME`.
    """
    name = cfg.MODEL.FPN_BLOCK.NAME
    return FPN_BLOCKS_REGISTRY.get(name)(cfg, input_shape)


def build_hed_head(cfg, input_shape):
    """
    Build an HED head from `cfg.MODEL.HED_HEAD.NAME`.
    """
    name = cfg.MODEL.HED_HEAD.NAME
    return HED_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_sem_seg_head(cfg, input_shape):
    """
    Build an semantic segmentation predictior from
     `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_semantic_instance_head(cfg, input_shape):
    """
    Build a joint semantic segmentation and contour predictior from
     `cfg.MODEL.SEMANTIC_INSTANCE_HEAD.NAME`.
    """
    name = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.NAME
    return SEMANTIC_INSTANCE_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_center_reg_head(cfg, input_shape):
    """
    Build an center regression predictior from
     `cfg.MODEL.CENTER_REG_HEAD.NAME`.
    """
    name = cfg.MODEL.CENTER_REG_HEAD.NAME
    return CENTER_REG_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_contour_head(cfg, input_shape):
    """
    Build an center regression predictior from
     `cfg.MODEL.CONTOUR_HEAD.NAME`.
    """
    name = cfg.MODEL.CONTOUR_HEAD.NAME
    return CONTOUR_HEAD_REGISTRY.get(name)(cfg, input_shape)


@FPN_BLOCKS_REGISTRY.register()
class FPNBlock(nn.Module):
    """
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

    def forward(self, features, targets=None):
        """
        Returns:
            Merged output from all layers of FPN
        """
        return self.layers(features)

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_blocks[i](features[f])
            else:
                x = x + self.scale_blocks[i](features[f])
        return x

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return ShapeSpec(channels=self.conv_dims, stride=self.common_stride)


@HED_HEAD_REGISTRY.register()
class HEDHead(nn.Module):
    """
    An HED (Holistic Edge Detection Network) Head that takes backbone inputs
     and generates generates contours. This is modified implementation adapted
     from HED and CASENet.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.HED_HEAD.IN_FEATURES
        n_feats = len(self.in_features)
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.loss_weight = cfg.MODEL.HED_HEAD.LOSS_WEIGHT
        norm = cfg.MODEL.HED_HEAD.NORM
        num_classes = cfg.MODEL.HED_HEAD.NUM_CLASSES
        self.num_classes = num_classes
        self.huber_active = cfg.MODEL.HED_HEAD.HUBER_ACTIVE
        self.dice_active = cfg.MODEL.HED_HEAD.DICE_ACTIVE
        self.res5_supervision = cfg.MODEL.HED_HEAD.RES5_SUPERVISION
        self.scale_blocks = []
        for in_feature in self.in_features:
            out_dims = 1
            activation = F.relu
            norm_module = get_norm(norm, out_dims)
            if in_feature == self.in_features[-1]:
                out_dims = num_classes
                if self.res5_supervision:
                    activation = None
                    norm_module = get_norm("", out_dims)

            # if in_feature == self.in_features[0]:
            block_stride = 2**max(1, int(np.log2(feature_strides[in_feature])))
            # else:
            #    block_stride = 1

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

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        fused, res5 = self.layers(features)
        if self.training:
            return fused, self.losses(fused, res5, targets)
        else:
            return fused, {}

    def layers(self, features):
        x = []
        for i, f in enumerate(self.in_features):
            x.append(self.scale_blocks[i](features[f]))
            if f == self.in_features[-1] and self.res5_supervision:
                res5 = x[i]
        x = self.shared_concat(x, self.num_classes)
        x = self.predictor(x)
        # print(torch.unique(torch.sigmoid(x[0])))
        # axis = plt.subplots(1, 2)[-1]
        # axis[0].imshow(rgb_from_pred_contours(x[0].squeeze().cpu().numpy()))
        # axis[1].imshow(rgb_from_pred_contours(
        #     x[0].squeeze().cpu().numpy(), thinned=True))
        # plt.show()
        if self.res5_supervision:
            return x, res5
        else:
            return x, None

    def losses(self, fused, res5, targets):
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
            # print('loss_fused:', loss_fused)
            # print('loss_res5:', loss_res5)
            # print('total_loss:', loss)
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

    def shared_concat(self, features, num_classes):
        n_feats = len(features)
        out_dim = num_classes * n_feats
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

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.predictor(features)
        if self.training:
            return x, self.losses(x, targets)
        else:
            return x, {}

    def losses(self, predictions, targets):
        x = F.interpolate(predictions, size=targets.size()[-2:],
                          mode="bilinear", align_corners=True)
        loss = F.cross_entropy(x, targets, reduction="mean",
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

    def forward(self, features, seg_targets, contour_targets):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.predictor(features)
        if self.training:
            return x, self.losses(x, seg_targets, contour_targets)
        else:
            return x, {}

    def losses(self, predictions, seg_targets, contour_targets):
        x = F.interpolate(predictions, size=seg_targets.size()[-2:],
                          mode="bilinear", align_corners=True)
        combined_targets = combine_seg_contour_targets(
            seg_targets, contour_targets)
        seg_loss = F.cross_entropy(x, combined_targets, reduction="mean",
                                   ignore_index=self.ignore_value)
        losses = {"loss_sem_seg": seg_loss * self.loss_weight}

        if self.dual_loss:
            contour_preds = x[:, 19:, ...]
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

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.predictor(features)
        if self.training:
            return x, self.losses(x, targets)
        else:
            return x, {}

    def losses(self, predictions, targets):
        x = F.interpolate(predictions, size=targets.size()[-2:],
                          mode="bilinear", align_corners=True)
        loss = WeightedMultiClassBinaryCrossEntropy(self.huber_active,
                                                    self.dice_active)(x, targets)
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
        norm = cfg.MODEL.CENTER_REG_HEAD.NORM
        feature_channels = input_shape.channels
        num_classes = 2
        stride = input_shape.stride
        kernel_size = cfg.MODEL.CENTER_REG_HEAD.CONV_KERNEL_SIZE
        self.num_upsamples = max(1, int(np.log2(stride)))
        self.loss_weight = cfg.MODEL.CENTER_REG_HEAD.LOSS_WEIGHT
        self.upsample_blocks = []
        ops = []
        for k in range(self.num_upsamples):
            in_channels = feature_channels if k == 0 else out_channels
            out_channels = int(feature_channels * (1 + (k + 1)/2))
            norm_module = nn.GroupNorm(
                32, out_channels) if norm == "GN" else None
            norm_ = norm_module if k != self.num_upsamples - 1 else None
            act_ = F.relu if k != self.num_upsamples - 1 else None
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int(kernel_size/2),
                norm=norm_,
                activation=act_
            )
            weight_init.c2_msra_fill(conv)
            ops.append(conv)
            ops.append(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=True)
            )
        self.upsample_blocks.append(nn.Sequential(*ops))
        self.add_module('upsample_block', self.upsample_blocks[-1])
        self.predictor = Conv2d(out_channels, num_classes,
                                kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """

        x = self.layers(features)
        if self.training:
            return x, self.losses(x, targets)
        else:
            return x, {}

    def layers(self, features):
        for i in range(self.num_upsamples):
            if i == 0:
                x = self.upsample_blocks[i](features)
            else:
                x = self.upsample_blocks[i](x)
        x = self.predictior(x)
        return x

    def losses(self, predictions, targets):
        loss = HuberLoss()(predictions, targets['vecs'], targets['mask'])
        losses = {"loss_center_reg": loss * self.loss_weight}
        return losses


class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

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
        super(FPN, self).__init__()

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
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {
            k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, input):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        x = [input[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2,
                                              mode="bilinear",
                                              align_corners=True)
            lateral_features = lateral_conv(features)
            # lateral_features = F.interpolate(lateral_features,
            #                                  size=top_down_features.size(
            #                                  )[-2:],
            #                                  mode="nearest")
            # print(lateral_features.shape,
            #       top_down_features.shape, prev_features.shape)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = input.get(
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


def get_gt_contours(targets, image_sizes, num_classes):
    batch_size = len(targets)
    contours = []
    for target_im, size in zip(targets, image_sizes):
        contour_im = np.zeros((num_classes, size[0], size[1]),
                              dtype=np.uint8).squeeze()
        img = np.zeros((size[0], size[1]))
        # no gt
        if not target_im.has('gt_masks'):
            contours.append(torch.from_numpy(contour_im))
            continue

        bboxes = target_im.gt_boxes.tensor
        labels = target_im.gt_classes
        masks = target_im.gt_masks

        for mask, bbox, label in zip(masks, bboxes, labels):
            bbox = bbox.cpu().numpy().astype(np.uint16)
            x1, y1, x2, y2 = bbox
            h, w = y2 - y1, x2 - x1
            merged_cnts = []
            # polygon = copy.deepcopy(polygon)
            # for p in polygon:
            #     p[0::2] = p[0::2] - bbox[0]
            #     p[1::2] = p[1::2] - bbox[1]
            # bit_mask = polygons_to_bitmask(polygon, h, w)
            # bit_mask = bit_mask.astype(np.uint8)
            bit_mask = mask.cpu().numpy().astype(np.uint8)[y1:y2, x1:x2]
            cnts, _ = cv2.findContours(bit_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                continue
            # elif len(cnts) == 1:
            #     merged_cnts = cnts
            # else:
            #     for cnt in cnts:
            #         merged_cnts.extend(cnt) 
            #     merged_cnts = [np.array(merged_cnts)]
            #     merged_cnts = np.sort(merged_cnts, axis=0)[::-1]
            if num_classes == 1:
                contour_im[y1:y2, x1:x2] = cv2.drawContours(
                    img[y1:y2, x1:x2], cnts, -1, 1, 2)
            else:
                contour_im[label, y1:y2, x1:x2] = cv2.drawContours(
                    img[y1:y2, x1:x2], cnts, -1, 1, 2)
        contours.append(torch.from_numpy(contour_im).float())
    return contours


def combine_seg_contour_targets(gt_seg, gt_contours):
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
