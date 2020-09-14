"""Postprocessing utilities."""
import copy

import cv2
import numpy as np
import torch
from detectron2.structures import Instances
from scipy import ndimage as nd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from torch.nn import functional as F


class ContourNetPostProcessor:
    """Post Processing for ContourNet."""

    def __init__(self, cfg):
        """"Initialize with cfg parameters."""
        self.contour_classes = cfg.MODEL.CONTOUR_HEAD.NUM_CLASSES
        self.output_height = cfg.MODEL.CONTOUR_NET.POSTPROC.OUTPUT_HEIGHT
        self.output_width = cfg.MODEL.CONTOUR_NET.POSTPROC.OUTPUT_WIDTH
        self.in_stride = cfg.MODEL.FPN_BLOCK.COMMON_STRIDE
        self.conf_thresh = cfg.MODEL.CONTOUR_NET.POSTPROC.CONTOUR_CONF_THRESH
        self.min_pixel_area = cfg.MODEL.CONTOUR_NET.POSTPROC.MIN_PIX_AREA
        self.refine_on = cfg.MODEL.CONTOUR_NET.REFINE.ENABLED
        self.refine_params = {
            'split_eps': cfg.MODEL.CONTOUR_NET.REFINE.SPLIT_EPS,
            'split_min_samples': cfg.MODEL.CONTOUR_NET.REFINE.SPLIT_MIN_SAMPLES,
            'split_sample_size_ratio': cfg.MODEL.CONTOUR_NET.REFINE.SPLIT_SAMPLE_SIZE_RATIO,
            'split_sample_max_size': cfg.MODEL.CONTOUR_NET.REFINE.SPLIT_SAMPLE_MAX_SIZE,
            'merge_eps': cfg.MODEL.CONTOUR_NET.REFINE.MERGE_EPS,
            'merge_min_samples': cfg.MODEL.CONTOUR_NET.REFINE.MERGE_MIN_SAMPLES,
        }
        self.panoptic_on = cfg.MODEL.CONTOUR_NET.COMBINE.ENABLED
        self.panoptic_params = {
            'overlap_threshold': cfg.MODEL.CONTOUR_NET.COMBINE.OVERLAP_THRESH,
            'stuff_area_limit': cfg.MODEL.CONTOUR_NET.COMBINE.STUFF_AREA_LIMIT,
            'instance_conf_thresh': cfg.MODEL.CONTOUR_NET.COMBINE.INSTANCES_CONFIDENCE_THRESH
        }

    def crop_resize(self, data, image_size):
        """Crop and Resize the input feature map."""
        if len(data.shape) == 3:
            data = data[:, : image_size[0], : image_size[1]]
            data = data.expand(1, -1, -1, -1)
        data = F.interpolate(
            data, size=(self.output_height, self.output_width), mode="bilinear", align_corners=True
        )
        return data.squeeze()

    def post_process_batch(self, network_output, image_sizes):
        """Process a given set of data per batch of images."""
        processed_results = []
        segms = network_output["sem_seg"]
        contours = network_output["contours"]
        offsets = network_output["offsets"]
        for segm, contour, offset, image_size in zip(
                segms, contours, offsets, image_sizes):
            raw_data = {
                "sem_seg": self.crop_resize(segm, image_size),
                "contours": self.crop_resize(contour, image_size),
                "offsets": self.crop_resize(offset, image_size)
            }
            processed_result = self.postprocess(raw_data, image_size)
            processed_results.append(processed_result)
            if self.panoptic_on:
                panoptic_result = self.get_panoptic(processed_result)
            processed_results[-1]["panoptic_seg"] = panoptic_result
        return processed_results

    def postprocess(self, raw_data, image_size):
        """Process a given set of data per image."""
        segm = raw_data["sem_seg"]
        offsets = raw_data["offsets"]
        contours = raw_data["contours"]
        segm_r = segm.argmax(dim=0).detach().cpu().numpy()
        mask = (segm_r >= 11).astype(np.uint8)
        offsets = offsets.detach().cpu().numpy()

        # pylint: disable=no-member
        contour_r = (torch.sigmoid(contours) >
                     self.conf_thresh).squeeze().int().cpu().numpy()
        result = {
            "sem_seg": segm_r,
            "contours": contour_r,
            "offsets": offsets,
            "masks": mask
        }

        instance_img = self.get_instance_img(result)
        result["instances"] = self.get_instances(instance_img, segm[11:])

        return result

    # pyltin: disable=too-many-instance-attributes,no-member
    def get_instance_img(self, result):
        """Get instance image from contours and segmentation output."""
        segm = result["sem_seg"]
        inst = np.zeros_like(segm)
        inst_from_seg = np.zeros_like(segm)

        for i in range(self.contour_classes):
            # pylint: disable=no-member
            if self.contour_classes == 1:
                inst_mask = result["masks"]
                contour_ = (result["contours"] == 1).astype(np.uint8)
            else:
                inst_mask = (segm == (11 + i)).astype(np.uint8)
                contour_ = (result["contours"][i] == 1).astype(np.uint8)

            diff = (1 - contour_)*inst_mask
            # pylint: disable=no-member
            _, labels = cv2.connectedComponents(diff.astype(np.uint8))
            # labels = fill_with_nearest_neighbour(labels, (labels == 0)) * inst_mask
            inst += (labels)
            # inst = fill_with_nearest_neighbour(inst, (inst == 0)) * inst_mask
            inst = self.remove_small_instances(inst, inst_mask, fill=True)

        if self.refine_on:
            inst = self.refine_instances(inst, result['offsets'])
            # Crop hood pixels and remove edge pixels over width.
            inst[1000:, :] = 0
            inst[:, :5] = 0
            inst[:, -5:] = 0
        # pylint: disable=no-member
        return torch.from_numpy(inst).squeeze()

    def get_instances(self, instance_img, semantic_results):
        """Get instance result from instance image."""
        semantic_results = F.softmax(semantic_results, dim=0)
        unique_instances = torch.unique(instance_img)
        output_height, output_width = semantic_results.size()[1:]
        n_instances = len(unique_instances)
        # pylint: disable=no-member
        pred_masks = torch.zeros(
            (n_instances,) + (output_height, output_width))
        # pylint: disable=no-member
        pred_classes = torch.zeros((n_instances,)).int()
        # pylint: disable=no-member
        scores = torch.zeros((n_instances,))
        for i, instance_id in enumerate(unique_instances):
            if i == 0:
                continue
            mask = (instance_img == instance_id)
            mask = mask.unsqueeze(0).unsqueeze(0).float()
            mask = F.interpolate(mask, size=(
                output_height, output_width), mode="bilinear", align_corners=True)
            mask = mask.squeeze().bool()
            labels = semantic_results.argmax(dim=0)
            labels = torch.flatten(labels[mask])
            label = torch.argmax(torch.bincount(labels))
            # pylint: disable=no-member
            area = torch.sum(mask)
            # pylint: disable=no-member
            score = torch.sum(semantic_results[label][mask])/area
            pred_classes[i] = label
            pred_masks[i] = mask
            scores[i] = score
        instances = Instances((output_height, output_width))
        instances.pred_masks = pred_masks
        instances.pred_classes = pred_classes
        instances.scores = scores

        return Instances((self.output_height, self.output_width),
                         **instances.get_fields())

    def remove_small_instances(self, inst, inst_mask, fill=False):
        """Remove instance with area less than threshold."""
        for i in np.unique(inst):
            if i == 0:
                continue
            mask_ = (inst == i).astype(np.uint8)
            area = np.sum(mask_)
            if area < self.min_pixel_area:
                inst[inst == i] = 0
        if fill:
            inst = fill_with_nearest_neighbour(inst, (inst == 0)) * inst_mask
        return inst

    def refine_instances(self, inst, offsets):
        """Merge and Split instances using offsets."""
        # print("Before Refine {}".format(np.unique(inst)))
        inst = self.split_instances(inst, offsets)
        # print("After Split {}".format(np.unique(inst)))
        inst = self.merge_instances(inst, offsets)
        # print("After Merge {}".format(np.unique(inst)))
        return inst

    def split_instances(self, inst, offsets):
        """Split instances using offsets."""
        g_w, g_h = np.meshgrid(
            np.arange(offsets.shape[2]), np.arange(offsets.shape[1]))
        np.random.seed(30)
        split_instance = copy.deepcopy(inst)
        n_instances = np.unique(inst)
        eps = self.refine_params['split_eps']
        min_samples = self.refine_params['split_min_samples']
        ratio = self.refine_params['split_sample_size_ratio']
        for i in np.unique(inst):
            if i == 0:
                continue
            mask_ = (inst == i).astype(np.uint8)
            area = np.sum(mask_)
            xsys = np.nonzero(mask_)
            c_x = (g_h - offsets[0])[xsys]
            c_y = (g_w - offsets[1])[xsys]

            # c_x, c_y = xsys[0] - off_x[xsys], xsys[1] - off_y[xsys]
            centroids = np.stack([c_x, c_y], axis=1)
            max_size = np.clip(centroids.shape[0]*ratio, 1, 20000)
            ratio = max_size/centroids.shape[0]
            idx = np.random.choice(
                [False, True], centroids.shape[0], p=[1-ratio, ratio])
            centroids_subset = centroids[idx, :]
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(
                centroids_subset)

            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ < 2:
                continue
            core_points = get_core_points(
                centroids_subset, labels, n_clusters_)

            dist = cdist(core_points, centroids)
            labels = np.argmin(dist, axis=0)
            for i in range(1, n_clusters_):
                instance_id = np.max(n_instances) + 1
                idx = np.where(labels == i)
                split_instance[xsys[0][idx], xsys[1][idx]] = instance_id
                n_instances = np.append(n_instances, [instance_id])

        return split_instance

    def merge_instances(self, inst, offsets):
        """Merge instances using offsets."""
        merge_inst = copy.deepcopy(inst)
        centroid_dict = compute_centroid_dict(inst, offsets)
        eps = self.refine_params['merge_eps']
        min_samples = self.refine_params['merge_min_samples']
        if not centroid_dict:
            return merge_inst
        instance_ids = list(centroid_dict.keys())
        if len(instance_ids) == 1:
            return merge_inst
        centroids = np.array(list(centroid_dict.values()))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
        labels = db.labels_
        seen = {}
        if labels[-1] == len(labels) + 1:
            return merge_inst
        for i, label in enumerate(labels):
            if label not in seen:
                seen[label] = instance_ids[i]
                continue
            merge_inst[inst == instance_ids[i]] = seen[label]
        return merge_inst

    # pylint: disable=too-many-locals
    def get_panoptic(self, results):
        """
        Implement a simple combining logic following
        "combine_semantic_and_instance_predictions.py" in panopticapi
        to produce panoptic segmentation outputs.
        """
        # pylint: disable=no-member
        segm = torch.from_numpy(results["sem_seg"])
        panoptic_seg = torch.zeros_like(segm, dtype=torch.int32)
        current_segment_id = 0
        segments_info = []
        # pylint: disable=no-member
        instance_masks = results['instances'].pred_masks.to(
            dtype=torch.bool, device=panoptic_seg.device)
        # Add instances one-by-one, check for overlaps with existing ones
        for inst_id in range(len(results['instances'])):
            mask = instance_masks[inst_id]  # H,W
            mask_area = mask.sum().item()

            if mask_area == 0:
                continue

            intersect = (mask > 0) & (panoptic_seg > 0)
            intersect_area = intersect.sum().item()

            if intersect_area * 1.0 / mask_area > self.panoptic_params['overlap_threshold']:
                continue

            if intersect_area > 0:
                mask = mask & (panoptic_seg == 0)

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": True,
                    "category_id": results['instances'].pred_classes[inst_id].item(),
                    "instance_id": inst_id,
                }
            )

        # Add semantic results to remaining empty areas
        semantic_labels = torch.unique(segm).cpu().tolist()
        for semantic_label in semantic_labels:
            mask = (segm == semantic_label) & (panoptic_seg == 0)
            mask_area = mask.sum().item()
            if mask_area < self.panoptic_params['stuff_area_limit']:
                continue

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": False,
                    "category_id": semantic_label,
                    "area": mask_area,
                }
            )

        return panoptic_seg, segments_info


def compute_centroid_dict(inst, offsets):
    """Compute centroid dictionary."""
    centroid_dict = {}
    g_w, g_h = np.meshgrid(
        np.arange(offsets.shape[2]), np.arange(offsets.shape[1]))
    for i in np.unique(inst):
        if i == 0:
            continue
        category_id = i//1000
        # offsets[:, inst != i] = 0
        mask_ = (inst == i).astype(np.uint8)
        offsets = offsets.astype(np.int32)

        area = np.sum(mask_)
        xsys = np.nonzero(mask_)
        off_x, off_y = offsets[0], offsets[1]
        c_x = np.mean((g_h - offsets[0])[xsys])
        c_y = np.mean((g_w - offsets[1])[xsys])
        centroid_dict[i] = [c_x, c_y]
    return centroid_dict


def fill_with_nearest_neighbour(data, invalid=None):
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


def get_core_points(centroids, labels, n_clusters):
    """Get core points from clusters."""

    core_points = np.zeros([n_clusters, 2])
    for i in range(n_clusters):
        core_points[i, :] = np.mean(
            centroids[np.where(labels == i)], axis=0)

    return core_points
