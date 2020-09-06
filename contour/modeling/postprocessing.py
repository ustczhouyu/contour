"""Postprocessing utilities."""
from torch.nn import functional as F
import numpy as np
import cv2
from scipy import ndimage as nd
import torch
from detectron2.structures import Instances


# pylint: disable=too-many-arguments
def contour_postprocess(seg_results, contour_results,
                        center_reg_results, img_size,
                        output_height, output_width,
                        num_classes, conf_thresh=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    seg_results = crop_resize(seg_results, img_size,
                              output_height, output_width)
    center_reg_results = crop_resize(center_reg_results, img_size,
                                     output_height, output_width)
    if len(contour_results.shape) < 3:
        contour_results = contour_results.unsqueeze(0).float()
    contour_results = crop_resize(contour_results, img_size,
                                  output_height, output_width)
    seg, instances, contours, offsets = get_instances(seg_results, contour_results,
                                                      center_reg_results,
                                                      num_classes, conf_thresh)
    instances = Instances((output_height, output_width),
                          **instances.get_fields())
    return seg, instances, contours, offsets


def crop_resize(result, img_size, output_height, output_width):
    """Crop and Resize the input feature map."""
    if len(result.shape) == 3:
        result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )
    return result.squeeze()


def get_instances(semantic_results, contour_results, center_reg_results, num_classes, conf_thresh):
    """Get instances from results."""
    seg = semantic_results.argmax(dim=0).detach().cpu().numpy()
    offsets = center_reg_results.detach().cpu().numpy()
    mask = (seg >= 11).astype(np.uint8)
    # pylint: disable=no-member
    contours = (torch.sigmoid(contour_results) >
                conf_thresh).squeeze().int().cpu().numpy()

    instance_img = get_instance_img(seg, mask, contours, num_classes)
    instances = get_instance_result_from_img(instance_img.squeeze(),
                                             semantic_results[11:])

    return seg, instances, contours, offsets


def get_instance_result_from_img(instance_img, semantic_results):
    """Get instance result from instance image."""
    semantic_results = F.softmax(semantic_results, dim=0)
    unique_instances = torch.unique(instance_img)[1:]
    n_instances = len(unique_instances)
    # pylint: disable=no-member
    pred_masks = torch.zeros((n_instances,) + instance_img.size())
    # pylint: disable=no-member
    pred_classes = torch.zeros((n_instances,)).int()
    # pylint: disable=no-member
    scores = torch.zeros((n_instances,))
    for i, instance_id in enumerate(unique_instances):
        mask = (instance_img == instance_id)
        # pylint: disable=no-member
        area = torch.sum(mask)
        label = int(instance_id // 1000 - 11)
        # pylint: disable=no-member
        score = torch.sum(semantic_results[label][mask])/area
        pred_classes[i] = label
        pred_masks[i] = mask
        scores[i] = score
    instances = Instances(instance_img.shape)
    instances.pred_masks = pred_masks
    instances.pred_classes = pred_classes
    instances.scores = scores
    return instances


# pyltin: disable=too-many-instance-attributes,no-member
def get_instance_img(seg, mask, contours, num_classes):
    """Get instance image from contours and segmentation output."""
    inst = np.zeros_like(seg)
    inst_from_seg = np.zeros_like(seg)

    for i in range(num_classes):
        # pylint: disable=no-member
        if num_classes == 1:
            _, inst_from_seg = cv2.connectedComponents(
                (seg >= 11).astype(np.uint8))
            inst_mask = (seg >= 11).astype(np.uint8)
            contour_ = (contours == 1).astype(np.uint8)
        else:
            _, inst_from_seg = cv2.connectedComponents(
                (seg == (11 + i)).astype(np.uint8))
            inst_mask = (seg == (11 + i)).astype(np.uint8)
            contour_ = (contours[i] == 1).astype(np.uint8)

        diff = inst_from_seg * inst_mask * (1 - contour_)
        # pylint: disable=no-member
        _, labels = cv2.connectedComponents(diff.astype(np.uint8))
        labels = fill(labels, (labels == 0)) * mask
        inst += (seg * inst_mask * 1000 + labels*inst_mask)

    for i in np.unique(inst):
        mask_ = (inst == i).astype(np.uint8)
        area = np.sum(mask_)
        if area < 500:
            inst[inst == i] = 0
    inst = fill(inst, (inst == 0)) * mask
    # pylint: disable=no-member
    return torch.from_numpy(inst)


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """

    if invalid is None:
        invalid = np.isnan(data)

    ind = nd.distance_transform_edt(
        invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def to_rgb(bw_im):
    """Convert instances to rgb."""
    instances = np.unique(bw_im)
    instances = instances[instances != 0]
    rgb_im = [np.zeros(bw_im.shape, dtype=int),
              np.zeros(bw_im.shape, dtype=int),
              np.zeros(bw_im.shape, dtype=int)]
    for instance in instances:
        color = np.random.randint(0, 255, size=(3))
        rgb_im[0][instance == bw_im] = color[0]
        rgb_im[1][instance == bw_im] = color[1]
        rgb_im[2][instance == bw_im] = color[2]
    return np.stack([rgb_im[0], rgb_im[1], rgb_im[2]], axis=-1)
