"""Postprocessing utilities."""
import copy

import cv2
import numpy as np
import torch
from detectron2.structures import Instances
from scipy import ndimage as nd
from torch.nn import functional as F

from sklearn.cluster import DBSCAN


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

    if len(contour_results.shape) < 3:
        contour_results = contour_results.unsqueeze(0).float()
    instances, contours, offsets = get_instances(seg_results, contour_results,
                                                 center_reg_results,
                                                 num_classes, conf_thresh,
                                                 img_size, output_height, output_width)
    instances = Instances((output_height, output_width),
                          **instances.get_fields())
    offsets = crop_resize(center_reg_results, img_size,
                          output_height, output_width)
    contours = crop_resize(contour_results, img_size,
                           output_height, output_width)
    return instances, contours, offsets


def crop_resize(result, img_size, output_height, output_width):
    """Crop and Resize the input feature map."""
    if len(result.shape) == 3:
        result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )
    return result.squeeze()


def get_instances(semantic_results, contour_results,
                  center_reg_results, num_classes, conf_thresh,
                  img_size, output_height, output_width):
    """Get instances from results."""
    seg = semantic_results.argmax(dim=0).detach().cpu().numpy()
    offsets = center_reg_results.detach().cpu().numpy()
    offsets_scale = (output_height//offsets.shape[1])
    mask = (seg >= 11).astype(np.uint8)
    # pylint: disable=no-member
    contours = (torch.sigmoid(contour_results) >
                conf_thresh).squeeze().int().cpu().numpy()

    instance_img = get_instance_img(
        seg, mask, contours, offsets/offsets_scale, num_classes)
    # instance_img = instance_img.unsqueeze(0).unsqueeze(0).float()
    # instance_img = F.interpolate(instance_img, size=(
    #     output_height, output_width), mode="nearest")
    semantic_results = crop_resize(
        semantic_results, img_size, output_height, output_width)
    instances = get_instance_result_from_img(
        instance_img.squeeze(), semantic_results[11:])

    return instances, contours, offsets


def get_instance_result_from_img(instance_img, semantic_results):
    """Get instance result from instance image."""
    semantic_results = F.softmax(semantic_results, dim=0)
    unique_instances = torch.unique(instance_img)[1:]
    output_height, output_width = semantic_results.size()[1:]
    n_instances = len(unique_instances)
    # pylint: disable=no-member
    pred_masks = torch.zeros((n_instances,) + (output_height, output_width))
    # pylint: disable=no-member
    pred_classes = torch.zeros((n_instances,)).int()
    # pylint: disable=no-member
    scores = torch.zeros((n_instances,))
    for i, instance_id in enumerate(unique_instances):
        mask = (instance_img == instance_id)
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=(
            output_height, output_width), mode="bilinear",  align_corners=False)
        mask = mask.squeeze().bool()
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
def get_instance_img(seg, mask, contours, offsets, num_classes):
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
        if area < 32:
            inst[inst == i] = 0
    inst = refine_instances(inst, offsets)
    inst = fill(inst, (inst == 0)) * mask
    inst[250:, :] = 0
    inst[:, :1] = 0
    inst[:, -2:] = 0
    # pylint: disable=no-member
    return torch.from_numpy(inst)


def refine_instances(inst, offsets):
    """Merge and Split instances using offsets."""
    inst = merge_instances(inst, offsets)
    inst, cat_count = clean_instance_ids(inst)
    inst = split_instances(inst, offsets, cat_count)
    return inst


def clean_instance_ids(inst):
    """Clean instance ids."""
    n_instances = np.unique(inst)
    cat_count = {}
    category_id, instance_id = 0, 0
    for i in n_instances:
        if i == 0:
            continue
        cat_id = i//1000
        if cat_id != category_id:
            cat_count[category_id] = instance_id
            category_id = cat_id
            instance_id = 1
        else:
            instance_id += 1
        inst_id = category_id*1000 + instance_id
        if i != inst_id:
            inst[(inst == i)] = inst_id
    cat_count[category_id] = instance_id
    return inst, cat_count


def split_instances(inst, offsets, cat_count):
    """Split instances using offsets."""
    n_instances = np.unique(inst)
    for i in np.unique(inst):
        if i == 0:
            continue
        category_id = i//1000
        mask_ = (inst == i).astype(np.uint8)
        area = np.sum(mask_)
        # DBSCAN hangs/crashes with large data.
        if area >= 10000:
            continue
        xsys = np.nonzero(mask_)
        off_x, off_y = offsets[0], offsets[1]
        c_x, c_y = xsys[0] - off_x[xsys], xsys[1] - off_y[xsys]
        centroids = np.stack([c_x, c_y], axis=1)
        db = DBSCAN(eps=5, min_samples=5).fit(centroids)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        for i in range(1, n_clusters_):
            idxs = np.argwhere(labels == i)
            cat_count[category_id] += 1
            instance_id = category_id*1000 + cat_count[category_id]
            x_s, y_s = xsys[0][idxs], xsys[1][idxs]
            inst[(x_s, y_s)] = instance_id

    return inst


def compute_centroid_dict(inst, offsets):
    """Compute centroid dictionary."""
    centroid_dict = {}
    for i in np.unique(inst):
        if i == 0:
            continue
        category_id = i//1000
        mask_ = (inst == i).astype(np.uint8)
        area = np.sum(mask_)
        xsys = np.nonzero(mask_)
        off_x, off_y = offsets[0], offsets[1]
        c_x, c_y = np.mean(xsys[0] - off_x[xsys]
                           ), np.mean(xsys[1] - off_y[xsys])
        centroid_dict[i] = [c_x, c_y]
    return centroid_dict


def merge_instances(inst, offsets):
    """Merge instances using offsets."""
    merge_inst = copy.deepcopy(inst)
    centroid_dict = compute_centroid_dict(inst, offsets)
    if not centroid_dict:
        return merge_inst
    instance_ids = list(centroid_dict.keys())
    if len(instance_ids) == 1:
        return merge_inst
    centroids = np.array(list(centroid_dict.values()))
    db = DBSCAN(eps=5, min_samples=1).fit(centroids)
    labels = db.labels_
    seen = {}

    for i, label in enumerate(labels):
        if label not in seen:
            seen[label] = instance_ids[i]
            continue
        merge_inst[inst == instance_ids[i]] = seen[label]

    return merge_inst


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
