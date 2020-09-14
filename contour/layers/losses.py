"""Loss functions package."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from robust_loss_pytorch import AdaptiveLossFunction


class WeightedMultiClassBinaryCrossEntropy(nn.Module):
    """Weighted Multi Class BCE.

    Optional huber and inverse dice loss terms.
    Refer to https://arxiv.org/pdf/1705.09759.pdf
    """

    def __init__(self, huber_active=False, inverse_dice=False):
        super().__init__()
        self.bce = WeightedBinaryCrossEntropy(huber_active, inverse_dice)
        self.huber_active = huber_active
        self.inverse_dice = inverse_dice

    # pylint: disable=arguments-differ
    def forward(self, _input, _target):
        mean_l1 = 0.0
        mean_bce = 0.0
        mean_inv_dice = 0.0
        if len(_target.shape) < 4:
            _target = _target.unsqueeze(1)
        num_classes = _target.shape[1]
        for i in range(num_classes):
            loss = self.bce(_input[:, i, ...], _target[:, i, ...])
            mean_bce += loss['loss_contour_bce']
        loss_dict = {'loss_contour_bce': mean_bce}
        if self.huber_active:
            mean_l1 += loss['loss_contour_huber']
            loss_dict.update({'loss_contour_huber':  mean_l1})
        elif self.inverse_dice:
            mean_inv_dice += loss['loss_contour_inv_dice']
            loss_dict.update({'loss_contour_inv_dice':  0.1*mean_inv_dice})
            loss_dict.update({'loss_contour_bce':  50*mean_bce})
        elif self.inverse_dice and self.huber_active:
            loss_dict.update({'loss_contour_huber':  50*mean_l1})
        return loss_dict


class WeightedBinaryCrossEntropy(nn.Module):
    """Weighted BCE."""

    def __init__(self, huber_active=False, inverse_dice=False):
        super().__init__()
        self.huber_active = huber_active
        self.inverse_dice = inverse_dice

    # pylint: disable=arguments-differ
    def forward(self, _input, _target):
        mean_bce = 0.0
        mean_l1 = 0.0
        mean_inv_dice = 0.0
        mask = get_weight_mask(_target.float())
        bce_loss = F.binary_cross_entropy_with_logits(_input.squeeze().float(),
                                                      _target.float(),
                                                      weight=mask,
                                                      reduction='mean')
        mean_bce += bce_loss
        loss_dict = {'loss_contour_bce': mean_bce}
        assert _input.squeeze().size() == _target.squeeze().size()
        if self.huber_active:
            # pylint: disable=no-member
            l1_loss = huber_loss(torch.sigmoid(_input.squeeze()),
                                 _target.squeeze(), delta=0.3)
            mean_l1 += l1_loss
            loss_dict.update({'loss_contour_huber':  mean_l1})
        if self.inverse_dice:
            # pylint: disable=no-member
            mean_inv_dice += inv_dice_loss(torch.sigmoid(_input.squeeze()),
                                           _target)
            loss_dict.update({'loss_contour_inv_dice':  mean_inv_dice})
        return loss_dict


class FocalLoss(nn.Module):
    """Focal Loss. Refer to https://arxiv.org/abs/1708.02002."""

    def __init__(self, ignore_index=255, gamma=1):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    # pylint: disable=arguments-differ
    def forward(self, _input, _target):
        _n, _c, _h, _w = _input.size()
        _input, _target = flatten_data(_input, _target)
        _target = _target.long()
        ce_loss = F.cross_entropy(_input, _target, weight=None,
                                  ignore_index=self.ignore_index.cuda(),
                                  reduction='none')
        loss = ((1 - F.softmax(_input, dim=1).permute(
            0, 2, 3, 1).contiguous().view(-1, _c))**self.gamma).squeeze() * ce_loss
        return loss.mean()


class DualityCELoss(nn.Module):
    """Duality CE Loss for Semantic Segmentation.

    Uses cross entropy and huber loss.
    Refer to https://arxiv.org/pdf/2004.07684.pdf"""

    def __init__(self, weights=None, ignore_index=255):
        super().__init__()
        self.weights = weights
        self.ignore_index = ignore_index

    # pylint: disable=arguments-differ
    def forward(self, _input, _target):
        _input, _target = flatten_data(_input, _target)
        _target = _target.long()
        ce_loss = F.cross_entropy(_input, _target, weight=self.weights.cuda(),
                                  ignore_index=self.ignore_index)
        # pylint:disable=no-member
        smooth_l1_loss = huber_loss((torch.argmax(_input, dim=1) == _input.size(1)).float(),
                                    (_target == 19).float(), delta=0.3)
        return ce_loss + 50*smooth_l1_loss


class DualityFocalLoss(nn.Module):
    """Duality focal Loss for Semantic Segmentation.

    Uses cross entropy, focal and huber loss.
    Refer to https://arxiv.org/pdf/2004.07684.pdf"""

    def __init__(self, ignore_index=255, gamma=1):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    # pylint: disable=arguments-differ
    def forward(self, _input, _target):
        _n, _c, _h, _w = _input.size()
        _input, _target = flatten_data(_input, _target)
        _target = _target.long()
        ce_loss = F.cross_entropy(_input, _target,
                                  weight=None,
                                  ignore_index=self.ignore_index,
                                  reduction='none')
        loss = ((1 - F.softmax(_input, dim=1).permute(
            0, 2, 3, 1).contiguous().view(-1, _c))**self.gamma).squeeze() * ce_loss
        # pylint: disable=no-member
        smooth_l1_loss = huber_loss((torch.argmax(_input, dim=1) == _input.size(1)).float(),
                                    (_target == _input.size(1)).float(),
                                    delta=0.3)
        return loss.mean() + 50*smooth_l1_loss


class BboxLoss(nn.Module):
    """Loss module for bounding boxes.

    Computes classification and offsets regression loss."""

    # pylint: disable=arguments-differ
    def forward(self, _input, _target, weight):
        loss = 0.0
        if isinstance(_input['class'], dict):
            for key, value in _input['class'].items():
                logits = {'class': value, 'offsets': _input['offsets'][key]}
                loss += key*bbox_loss_level(logits, _target, weight, key)
        else:
            loss = bbox_loss_level(_input, _target, weight)

        return loss


class HuberLoss(nn.Module):
    """Huber loss (or) Smooth L1 loss."""

    def __init__(self, delta=0.3):
        super().__init__()
        self.delta = delta

    # pylint: disable=arguments-differ
    def forward(self, _input, _target, weight=None):
        if _input.size() != _target.size():
            _input = _input.permute(0, 2, 3, 1).squeeze()
        # pylint: disable=no-member
        abs_diff = torch.abs(_input - _target)
        cond = abs_diff < self.delta
        # pylint: disable=no-member
        loss = torch.where(cond, 0.5 * abs_diff ** 2,
                           (self.delta*abs_diff - 0.5*self.delta**2))
        if weight is not None:
            loss = loss * weight.unsqueeze(1)
        return loss.mean()


class AdaptiveLoss(nn.Module):
    """General and Adaptive Robust Loss Function.

    Refer to https://arxiv.org/pdf/1701.03077.pdf    
    """

    def __init__(self, num_dims=1, float_dtype=np.float32, device='cuda'):
        super().__init__()
        self.adaptive = AdaptiveLossFunction(num_dims, float_dtype, device)

    # pylint: disable=arguments-differ
    def forward(self, _input, _target, weight=None):
        if _input.size() != _target.size():
            _input = _input.permute(0, 2, 3, 1).squeeze()
        # pylint: disable=no-member
        abs_diff = torch.abs(_input - _target)
        if weight is not None:
            abs_diff = abs_diff * weight.unsqueeze(1)
        # pylint: disable=no-member
        abs_diff = torch.flatten(abs_diff)
        loss = torch.mean(self.adaptive.lossfun(abs_diff[:, None]))
        return loss


def huber_loss(_input, _target, weight=None, delta=0.5):
    """Compute Huber loss."""
    # pylint: disable=no-member
    abs_diff = torch.abs(torch.flatten(_input) - torch.flatten(_target))
    cond = abs_diff <= delta
    loss = torch.where(cond, 0.5 * abs_diff ** 2,
                       (delta*abs_diff - 0.5*delta**2))
    if weight is not None:
        loss = loss * weight.unsqueeze(1)
    return loss.mean()


def inv_dice_loss(_input, _target):
    """Compute Inverse dice loss."""
    # pylint: disable=no-member
    intersection = 2*torch.sum(_input*_target)
    union = torch.sum(_input**2) + torch.sum(_target**2)
    loss = -torch.log(intersection/union + 1e-6)
    return loss


def bbox_loss_level(_input, _target, weight, stride=1):
    """Bbox loss at a given level/stride."""
    class_target = F.max_pool2d(_target['class'], kernel_size=stride)
    offset_target = F.interpolate(_target['offsets'], scale_factor=1/stride,
                                  mode='bilinear', align_corners=True)
    weight = F.max_pool2d(weight, kernel_size=stride)
    class_loss = focal_loss(_input['class'], class_target, size_average=False)
    class_loss = (class_loss*weight.view(-1)).mean()
    bbox_loss = huber_loss(
        _input['offsets'], offset_target, weight=weight)
    return class_loss + bbox_loss


def focal_loss(_input, _target, gamma=1, size_average=True):
    """Compute focal loss."""
    num_classes = _input.size()[1]
    softmax_preds = F.softmax(_input, dim=1).permute(
        0, 2, 3, 1).contiguous().view(-1, num_classes)
    _input, _target = flatten_data(_input, _target)
    _target = _target.long()
    ce_loss = F.cross_entropy(_input, _target, weight=None,
                              ignore_index=255, reduction='none')
    _target = _target * (_target != 255).long()
    # pylint: disable=no-member
    softmax_preds = torch.gather(softmax_preds, 1, _target.unsqueeze(1))
    loss = ((1 - softmax_preds) ** gamma).squeeze() * ce_loss
    if size_average:
        return loss.mean()
    return loss


def flatten_data(_input, _target):
    """Flatten tensor to 1 dimension."""
    _n, _c, _h, _w = _input.size()
    _input = _input.squeeze()
    _target = _target.squeeze()
    if len(_target.size()) == 2 and _n == 1:
        _ht, _wt = _target.size()
        _input = _input.transpose(0, 1).transpose(1, 2)
    elif len(_target.size()) == 3 and _n > 1:
        _, _ht, _wt = _target.size()
        _input = _input.transpose(1, 2).transpose(2, 3)
    else:
        raise ValueError('Check size of _inputs and targets')

    # Handle inconsistent size between _input and _target
    if _h != _ht and _w != _wt:  # upsample labels
        _input = F.interpolate(_input, size=(_ht, _wt),
                               mode="bilinear", align_corners=True)
    _input = _input.contiguous().view(-1, _c)
    _target = _target.view(-1)

    return _input, _target


def get_weight_mask(label):
    """Get weight mask from labels."""
    # pylint: disable=no-member
    mask = torch.zeros_like(label)
    num_el = label.numel()
    # pylint: disable=no-member
    beta = torch.sum((label == 0).float()) / num_el
    mask[label != 0] = beta
    mask[label == 0] = 1.0 - beta
    return mask


def make_one_hot(labels, num_classes=10):
    """Get one hot labels."""
    _n, _h, _w = labels.size()
    # pylint: disable=no-member
    one_hot = torch.zeros((_n, num_classes, _h, _w), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id, ...] = (labels == class_id+1)
    return one_hot.cuda()
