import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d, spatial_gradient

EPS = 1e-6


class StealLoss(nn.Module):
    def __init__(self, length=2, tau=0.1):
        super().__init__()
        self.r_ = length
        self.tau = tau
        # Get kernels that represent normal directions.
        self.filter_dict = get_filter_dict(length)

    def forward(self, predictions, labels):
        """Calculate NMS loss and direction loss."""
        # Find edge directions
        true_angles = get_normal_anlges(labels.unsqueeze(1))
        pred_angles = get_normal_anlges(predictions)
        # dir_loss = F.l1_loss(pred_angles, true_angles)
        # loss_dict = {'loss_contour_dir': dir_loss}
        loss_dict = {}

        exp_preds = torch.exp(predictions/self.tau)
        nms_loss = -1*torch.mean(self._get_nms_loss(exp_preds, true_angles))
        loss_dict['loss_contour_nms'] = nms_loss

        return loss_dict

    def _get_nms_loss(self, exp_preds, angles):
        loss = 0.0
        loss += self._get_nms_from_section(
            exp_preds, angles, section='horizontal')
        loss += self._get_nms_from_section(
            exp_preds, angles, section='vertical')
        loss += self._get_nms_from_section(
            exp_preds, angles, section='lead_diag')
        loss += self._get_nms_from_section(
            exp_preds, angles, section='cnt_diag')

        return loss

    def _get_nms_from_section(self, exp_preds, angles, section):
        norm = self._get_softmax_response(exp_preds, section)
        mask = get_mask_from_section(angles, section)

        # Get direction NMS loss and mask it.
        return (torch.log(norm) * mask).unsqueeze_(1)

    def _get_softmax_response(self, exp_preds, section='horizontal', eps=EPS):
        kernel = self.filter_dict[section].to(exp_preds.device)
        boundary_responses_sum = F.conv2d(exp_preds, kernel, padding=self.r_)
        norm = torch.div(exp_preds, boundary_responses_sum + eps)

        return norm


class DirectionLoss(nn.Module):
    def forward(self, predictions, labels):
        """Calculate direction loss."""
        # Find edge directions
        true_angles = get_normal_anlges(labels)
        pred_angles = get_normal_anlges(predictions)
        dir_loss = F.l1_loss(pred_angles, true_angles)
        loss_dict = {'loss_contour_dir': dir_loss}

        return loss_dict


class NMSLayer(nn.Module):
    def __init__(self, length=1):
        super.__init__()
        self.r_ = length
        # Get kernels that represent normal directions.
        self.filter_dict = get_filter_dict(length)

    def forward(self, predictions):
        """Supress the predictions along normal."""
        # Find edge directions
        pred_angles = get_normal_anlges(predictions)

        # TODO: output predictions after thinning.

        return predictions


def get_filter_dict(r_=2):
    filter_dict = {}
    horiz = torch.zeros((1, 1, 2 * r_ + 1, 2 * r_ + 1))
    horiz[:, :, r_, :] = 1
    filter_dict['horizontal'] = horiz
    vert = torch.zeros((1, 1, 2 * r_ + 1, 2 * r_ + 1))
    vert[:, :, :, r_] = 1
    filter_dict['vertical'] = vert
    filter_dict['cnt_diag'] = torch.eye(2*r_+1).unsqueeze(0).unsqueeze(0)
    lead_diag = np.array(np.fliplr(np.eye(2*r_+1)))
    filter_dict['lead_diag'] = torch.tensor(
        lead_diag).unsqueeze(0).unsqueeze(0).float()
    return filter_dict


def get_mask_from_section(angles, section='horizontal'):
    if section == 'horizontal':
        return get_mask(angles, -np.pi/8, np.pi/8).float()
    elif section == 'lead_diag':
        return get_mask(angles, np.pi/8, 3*(np.pi/8)).float()
    elif section == 'cnt_diag':
        return get_mask(angles, -3*(np.pi/8), -np.pi/8).float()
    elif section == 'vertical':
        return (get_mask(angles, 3*(np.pi/8), np.pi/2) |
                get_mask(angles, -np.pi/2, -3*(np.pi/8))).float()


def get_mask(angles, start, end):
    return (angles >= start) & (angles < end)


def get_normal_anlges(image, eps=EPS):
    """Calculate the normal direction of edges.

    Ref: https://github.com/nv-tlabs/STEAL/blob/master/utils/edges_nms.m
    """
    first_grads = spatial_gradient(gaussian_blur2d(image, (5, 5), (2, 2)))
    second_grad_x = spatial_gradient(
        first_grads[:, :, 0, :, :].squeeze_(2))
    second_grad_y = spatial_gradient(
        first_grads[:, :, 1, :, :].squeeze_(2))

    grad_xx = second_grad_x[:, :, 0, :, :].squeeze_()
    grad_xy = second_grad_y[:, :, 0, :, :].squeeze_()
    grad_yy = second_grad_y[:, :, 1, :, :].squeeze_()
    angle = torch.atan(
        grad_yy * torch.sign(-(grad_xy + eps)) / (grad_xx + eps))
    return angle


if __name__ == "__main__":
    np.random.seed(10)
    loss_fn = StealLoss()
    pred = torch.rand(1, 1, 3, 3)
    labels = torch.rand(1, 1, 3, 3)
    loss = loss_fn(pred, labels)
