"""Visualization utlities."""
import cv2
import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer as DetectronVideoVisualizer


class ImageVisualizer(Visualizer):
    """Image Visualizer with support to draw contours and offsets."""
    def draw_contours(self, contours, data_type='predictions', alpha=0.8):
        """
        Args:
            contours (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        if isinstance(contours, torch.Tensor):
            contours = contours.numpy()
        if data_type == 'predictions':
            rgb = rgb_from_pred_contours(contours, self.metadata)
        elif data_type == 'ground_truth':
            rgb = rgb_from_gt_contours(contours, self.metadata)
        shape2d = (rgb.shape[0], rgb.shape[1])
        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = rgb
        # pylint: disable=no-member
        binary = cv2.cvtColor((rgb*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        rgba[:, :, 3] = (binary != 0).astype("float32") * alpha
        self.output.ax.imshow(rgba)
        # print(np.unique(rgb))
        return self.output

    def draw_offsets(self, offsets, alpha=0.8):
        """
        Args:
            offsets (ndarray): numpy array of shape (2, H, W), where H is the image height and
                W is the image width. Each value in the array is float32 represting offset to
                centroid.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.numpy()

        rgb = np.zeros((offsets.shape[-2], offsets.shape[-1], 3))
        rgb[:, :, 0] = offsets[0, ...]
        rgb[:, :, 1] = offsets[1, ...]
        rgb = rgb - np.min(rgb)
        rgb = rgb / np.max(rgb)
        shape2d = (rgb.shape[0], rgb.shape[1])
        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = rgb
        # pylint: disable=no-member
        binary = cv2.cvtColor((rgb*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        rgba[:, :, 3] = (binary != 0).astype("float32") * alpha
        self.output.ax.imshow(rgba)
        return self.output


def rgb_from_pred_contours(score_fuse_feats, metadata=None, conf_thresh=0.5):
    """ Generate rgb visuals from contour_predictions
    score_fuse_feats = torch.sigmoid(score_fuse_feats[0].squeeze()).cpu().numpy()
    Arguments:
        score_fuse_feats {[type]} -- Sigmoid features [cxhxw]
    """
    if len(score_fuse_feats.shape) < 3:
        score_fuse_feats = np.expand_dims(score_fuse_feats, axis=0)
    num_classes = score_fuse_feats.shape[0]
    height, width = score_fuse_feats.shape[1], score_fuse_feats.shape[2]
    _r = np.zeros((height, width))
    _g = np.zeros((height, width))
    _b = np.zeros((height, width))
    rgb = np.zeros((height, width, 3))
    if metadata is not None:
        color_dict = metadata.thing_colors
    for idx_cls in range(num_classes):
        score_pred = score_fuse_feats[idx_cls, ...]
        score_pred_flag = (score_pred > conf_thresh).astype(np.uint8)
        _r[score_pred_flag == 1] = color_dict[idx_cls][0]
        _g[score_pred_flag == 1] = color_dict[idx_cls][1]
        _b[score_pred_flag == 1] = color_dict[idx_cls][2]
    rgb[:, :, 0] = _r/255.0
    rgb[:, :, 1] = _g/255.0
    rgb[:, :, 2] = _b/255.0

    return rgb


def rgb_from_gt_contours(gt_data, metadata=None):
    """Generate RGB Visual from gt_contours.

    Arguments:
        gt_data {[type]} -- [cxhxw]
    """
    if len(gt_data.shape) < 3:
        gt_data = np.expand_dims(gt_data, axis=0)
    num_classes = gt_data.shape[0]
    height, width = gt_data.shape[1], gt_data.shape[2]
    _r = np.zeros((height, width))
    _g = np.zeros((height, width))
    _b = np.zeros((height, width))
    rgb = np.zeros((height, width, 3))
    if metadata is not None:
        color_dict = metadata.thing_colors
    for idx_cls in range(num_classes):
        score_pred_flag = gt_data[idx_cls, ...]
        _r[score_pred_flag == 1] = color_dict[idx_cls][0]
        _g[score_pred_flag == 1] = color_dict[idx_cls][1]
        _b[score_pred_flag == 1] = color_dict[idx_cls][2]
    rgb[:, :, 0] = _r/255.0
    rgb[:, :, 1] = _g/255.0
    rgb[:, :, 2] = _b/255.0
    return rgb


class VideoVisualizer(DetectronVideoVisualizer):
    """Video Visualizer with support to draw contours and offsets."""
    pass
