from torch.nn import functional as F
import numpy as np
import cv2
from scipy import ndimage as nd
import torch
from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom
# import matplotlib.pyplot as plt


def contour_postprocess(seg_results, contour_results,
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
    # size = (output_height, output_width)
    # seg_results = F.interpolate(
    #     seg_results.unsqueeze(0), size=size, mode='bilinear').squeeze()
    contour_results = F.interpolate(contour_results.unsqueeze(0),
                                    size=(output_height, output_width),
                                    mode="bilinear",
                                    align_corners=True).squeeze()
    results = get_instances(seg_results, contour_results,
                            num_classes, conf_thresh)
    results = Instances((output_height, output_width), **results.get_fields())
    return results


def get_instances(semantic_results, contour_results, num_classes, conf_thresh):
    seg = semantic_results.argmax(dim=0).cpu().numpy()
    mask = (seg >= 11).astype(np.uint8)
    # if num_classes == 1:
    contours = (torch.sigmoid(contour_results) >
                conf_thresh).squeeze().int().cpu().numpy()
    # else:
    #     contours = contour_results.argmax(dim=0).cpu().numpy()
    instance_img = get_instance_from_contour(seg, mask, contours, num_classes)
    instances = get_instance_result(instance_img.squeeze(),
                                    semantic_results[11:])

    return instances


def get_instance_result(instance_img, semantic_results):
    semantic_results = F.softmax(semantic_results, dim=0)
    unique_instances = torch.unique(instance_img)[1:]
    n_instances = len(unique_instances)
    pred_masks = torch.zeros((n_instances,) + instance_img.size())
    pred_classes = torch.zeros((n_instances,)).int()
    scores = torch.zeros((n_instances,))
    semantic_maps = {i: semantic_map for i,
                     semantic_map in enumerate(semantic_results)}
    for i, instance_id in enumerate(unique_instances):
        mask = (instance_img == instance_id)
        area = torch.sum(mask)
        label = int(instance_id // 1000 - 11)
        score = torch.sum(semantic_maps[label][mask])/area
        pred_classes[i] = label
        pred_masks[i] = mask
        scores[i] = score
    instances = Instances(instance_img.shape)
    instances.pred_masks = pred_masks
    instances.pred_classes = pred_classes
    instances.scores = scores
    return instances


def get_instance_from_contour(seg, mask, contours, num_classes):
    inst = np.zeros_like(seg)
    inst_from_seg = np.zeros_like(seg)
    for i in np.unique(seg):
        _, inst_ = cv2.connectedComponents(((seg == i)*mask).astype(np.uint8))
        inst_from_seg += inst_
    for i in range(num_classes):
        # if i == 0:
        #     continue
        kernel = np.ones((2, 2), np.uint8)
        if num_classes == 1:
            inst_mask = (seg >= 11).astype(np.uint8)
            contour_ = (contours == 1).astype(np.uint8)
        else:
            inst_mask = (seg == (11 + i)).astype(np.uint8)
            contour_ = (contours[i] == 1).astype(np.uint8)

        contour = cv2.morphologyEx(contour_, cv2.MORPH_CLOSE, kernel)
        diff = inst_from_seg * inst_mask * (1 - contour)
        _, labels = cv2.connectedComponents(diff.astype(np.uint8))
        inst += (seg * inst_mask * (1 - contour) * 1000 + labels)
    for i in np.unique(inst):
        mask_ = (inst == i).astype(np.uint8)
        area = np.sum(mask_)
        if area < 100:
            inst[inst == i] = 0
    # axis = plt.subplots(4, 2)[-1]
    # axis[0, 0].imshow(to_rgb(seg))
    # axis[0, 1].imshow(to_rgb(diff))
    # axis[1, 0].imshow(mask)
    # axis[1, 1].imshow(to_rgb(inst_from_seg))
    # axis[2, 0].imshow(contour_)
    # axis[2, 1].imshow(contour)
    # axis[3, 0].imshow(to_rgb(inst))
    inst = fill(inst, (inst == 0)) * mask
    # axis[3, 1].imshow(to_rgb(inst))
    # plt.show()
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
