import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedMultiClassBinaryCrossEntropy(nn.Module):

    def __init__(self, huber_active=False, inverse_dice=False):
        super(WeightedMultiClassBinaryCrossEntropy, self).__init__()
        self.bce = WeightedBinaryCrossEntropy(huber_active, inverse_dice)
        self.huber_active = huber_active
        self.inverse_dice = inverse_dice

    def forward(self, input, target):
        mean_l1 = 0.0
        mean_bce = 0.0
        mean_inv_dice = 0.0
        if len(target.shape) < 4:
            target = target.unsqueeze(1)
        num_classes = target.shape[1]
        for i in range(num_classes):
            loss = self.bce(input[:, i, ...], target[:, i, ...])
            mean_bce += loss['loss_hed_bce']
        loss_dict = {'loss_hed_bce': mean_bce}
        if self.huber_active:
            mean_l1 += loss['loss_hed_huber']
            loss_dict.update({'loss_hed_huber':  mean_l1})
        if self.inverse_dice:
            mean_inv_dice += loss['loss_hed_inv_dice']
            loss_dict.update({'loss_hed_inv_dice':  0.1*mean_inv_dice})
            loss_dict.update({'loss_hed_bce':  50*mean_bce})
        if self.inverse_dice and self.huber_active:
            loss_dict.update({'loss_hed_huber':  50*mean_l1})
        return loss_dict


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, huber_active=False, inverse_dice=False):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.huber_active = huber_active
        self.inverse_dice = inverse_dice

    def forward(self, input, target):
        n_batch = input.shape[0]
        mean_bce = 0.0
        mean_l1 = 0.0
        mean_inv_dice = 0.0
        mask = get_weight_mask(target.float())
        bce_loss = F.binary_cross_entropy_with_logits(input.squeeze().float(),
                                                      target.float(),
                                                      weight=mask,
                                                      reduction='mean')
        mean_bce += bce_loss
        loss_dict = {'loss_hed_bce': mean_bce}
        assert input.squeeze().size() == target.squeeze().size()
        if self.huber_active:
            l1_loss = huber_loss(torch.sigmoid(input.squeeze()),
                                 target.squeeze(), delta=0.3)
            mean_l1 += l1_loss
            loss_dict.update({'loss_hed_huber':  mean_l1})
        if self.inverse_dice:
            mean_inv_dice += inv_dice_loss(torch.sigmoid(input.squeeze()),
                                           target)
            loss_dict.update({'loss_hed_inv_dice':  mean_inv_dice})
        return loss_dict


class FocalLoss(nn.Module):

    def __init__(self, ignore_index=255, gamma=1):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, input, target):
        n, c, h, w = input.size()
        input, target = flatten_data(input, target)
        target = target.long()
        ce_loss = F.cross_entropy(input, target, weight=None,
                                  ignore_index=self.ignore_index.cuda(),
                                  reduction='none')
        focal_loss = ((1 - F.softmax(input, dim=1).permute(
            0, 2, 3, 1).contiguous().view(-1, c))**self.gamma).squeeze() * ce_loss
        return focal_loss.mean()


class DualityCELoss(nn.Module):

    def __init__(self, weights=None, ignore_index=255):
        super(DualityCELoss, self).__init__()
        self.weights = weights
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input, target = flatten_data(input, target)
        target = target.long()
        ce_loss = F.cross_entropy(input, target, weight=self.weights.cuda(),
                                  ignore_index=self.ignore_index)
        smooth_l1_loss = huber_loss((torch.argmax(input, dim=1) == input.size(1)).float(),
                                    (target == 19).float(), delta=0.3)
        return ce_loss + 50*smooth_l1_loss


class DualityFocalLoss(nn.Module):

    def __init__(self, ignore_index=255, gamma=1):
        super(DualityFocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, input, target):
        n, c, h, w = input.size()
        input, target = flatten_data(input, target)
        target = target.long()
        ce_loss = F.cross_entropy(input, target,
                                  weight=None,
                                  ignore_index=self.ignore_index,
                                  reduction='none')
        focal_loss = ((1 - F.softmax(input, dim=1).permute(
            0, 2, 3, 1).contiguous().view(-1, c))**self.gamma).squeeze() * ce_loss

        smooth_l1_loss = huber_loss((torch.argmax(input, dim=1) == input.size(1)).float(),
                                    (target == input.size(1)).float(),
                                    delta=0.3)
        return focal_loss.mean() + 50*smooth_l1_loss


class BboxLoss(nn.Module):

    def __init__(self):
        super(BboxLoss, self).__init__()

    def forward(self, input, target, weight):
        loss = 0.0
        if isinstance(input['class'], dict):
            for k, v in input['class'].items():
                logits = {'class': v, 'offsets': input['offsets'][k]}
                loss += k*bbox_loss_level(logits, target, weight, k)
        else:
            loss = bbox_loss_level(input, target, weight)

        return loss


class HuberLoss(nn.Module):

    def __init__(self, delta=0.3):
        super(HuberLoss, self).__init__()
        self.deta = delta

    def forward(self, input, target, weight):
        deta = self.deta.cuda()
        if input.size() != target.size():
            input = input.permute(0, 2, 3, 1).squeeze()
        abs_diff = torch.abs(input - target)
        cond = abs_diff < self.delta
        loss = torch.where(cond, 0.5 * abs_diff ** 2,
                           (self.delta*abs_diff - 0.5*self.delta**2))
        if weight is not None:
            loss = loss * weight.unsqueeze(1)
        return loss.mean()


def huber_loss(input, target, weight=None, delta=0.5, size_average=True):
    abs_diff = torch.abs(input - target)
    cond = abs_diff <= delta
    loss = torch.where(cond, 0.5 * abs_diff ** 2,
                       (delta*abs_diff - 0.5*delta**2))
    if weight is not None:
        loss = loss * weight.unsqueeze(1)
    return loss.mean()

def inv_dice_loss(input, target):
    intersection = 2*torch.sum(input*target)
    union = torch.sum(input**2) + torch.sum(target**2)
    loss = -torch.log(intersection/union + 1e-6)
    return loss


def bbox_loss_level(input, target, weight, stride=1):
    class_target = F.max_pool2d(target['class'], kernel_size=stride)
    offset_target = F.interpolate(target['offsets'], scale_factor=1/stride,
                                  mode='bilinear', align_corners=True)
    weight = F.max_pool2d(weight, kernel_size=stride)
    class_loss = focal_loss(input['class'], class_target, size_average=False)
    class_loss = (class_loss*weight.view(-1)).mean()
    bbox_loss = huber_loss(
        input['offsets'], offset_target, weight=weight)
    return class_loss + bbox_loss


def focal_loss(input, target, weights=None, sample_weight=None,
               gamma=1, size_average=True):
    c = input.size()[1]
    softmax_preds = F.softmax(input, dim=1).permute(
        0, 2, 3, 1).contiguous().view(-1, c)
    input, target = flatten_data(input, target)
    target = target.long()
    ce_loss = F.cross_entropy(input, target, weight=None,
                              ignore_index=255, reduction='none')
    target = target * (target != 255).long()
    softmax_preds = torch.gather(softmax_preds, 1, target.unsqueeze(1))
    focal_loss = ((1 - softmax_preds) ** gamma).squeeze() * ce_loss
    if size_average:
        return focal_loss.mean()
    else:
        return focal_loss


def flatten_data(input, target):
    n, c, h, w = input.size()
    input = input.squeeze()
    target = target.squeeze()
    if len(target.size()) == 2 and n == 1:
        ht, wt = target.size()
        input = input.transpose(0, 1).transpose(1, 2)
    elif len(target.size()) == 3 and n > 1:
        nt, ht, wt = target.size()
        input = input.transpose(1, 2).transpose(2, 3)
    else:
        raise ValueError('Check size of inputs and targets')

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt),
                              mode="bilinear", align_corners=True)
    input = input.contiguous().view(-1, c)
    target = target.view(-1)

    return input, target


def get_weight_mask(label):
    mask = torch.zeros_like(label)
    num_el = label.numel()
    beta = torch.sum((label == 0).float()) / num_el
    mask[label != 0] = beta
    mask[label == 0] = 1.0 - beta
    return mask


def make_one_hot(labels, num_classes=10):
    n, h, w = labels.size()
    one_hot = torch.zeros((n, num_classes, h, w), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id, ...] = (labels == class_id+1)
    return one_hot.cuda()
