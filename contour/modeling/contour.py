import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_roi_heads, roi_heads
from detectron2.modeling.backbone import build_backbone, FPN
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from torch import nn

from .utils import (build_fpn, build_fpn_block,
                    build_sem_seg_head, build_hed_head, get_gt_contours)
from .postprocessing import contour_postprocess

__all__ = ["ContourNet"]


@META_ARCH_REGISTRY.register()
class ContourNet(nn.Module):
    """
    Meta Architecture for ContourNet that outputs instance or panoptic seg
     using contours.

               |-- fpn -- fpn_block -- sem_seg_head --|
    Backbone --
               |-- hed_head --------------------------|-- postproc -- instance
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.fpn = build_fpn(cfg, self.backbone.output_shape())
        self.fpn_block = build_fpn_block(cfg, self.fpn.output_shape())
        self.sem_seg_head = build_sem_seg_head(cfg,
                                               self.fpn_block.output_shape())
        self.hed_head = build_hed_head(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(
            cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(
            cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.combine_on = cfg.MODEL.CONTOUR_NET.COMBINE.ENABLED
        self.loss_combination = cfg.MODEL.CONTOUR_NET.LOSS_COMBINATION
        if self.loss_combination == 'uncertainty':
            self.sigma = nn.Parameter(torch.ones(3))
        else:
            loss_weights = []
            loss_weights.append(cfg.MODEL.CENTER_REG_HEAD.LOSS_WEIGHT)
            loss_weights.append(cfg.MODEL.HED_HEAD.LOSS_WEIGHT)
            loss_weights.append(cfg.MODEL.HED_HEAD.LOSS_WEIGHT)
            self.sigma = torch.tensor(loss_weights)
        self.combine_overlap_threshold = \
            cfg.MODEL.CONTOUR_NET.COMBINE.OVERLAP_THRESH
        self.combine_stuff_area_limit = \
            cfg.MODEL.CONTOUR_NET.COMBINE.STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = \
            cfg.MODEL.CONTOUR_NET.COMBINE.INSTANCES_CONFIDENCE_THRESH
        self.contour_classes = cfg.MODEL.HED_HEAD.NUM_CLASSES

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        fpn_features = self.fpn_block(self.fpn(features))

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(gt_sem_seg,
                                                self.backbone.size_divisibility,
                                                self.sem_seg_head.ignore_value
                                                ).tensor
        else:
            gt_sem_seg = None

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
            gt_contours = get_gt_contours(gt_instances, images.image_sizes,
                                          self.contour_classes)
        else:
            gt_instances = None
            gt_contours = None

        sem_seg_results, sem_seg_losses = self.sem_seg_head(fpn_features,
                                                            gt_sem_seg)
        contour_results, contour_losses = self.hed_head(features, gt_contours)

        if self.training:
            losses = {}
            losses.update(sem_seg_losses)
            losses.update(contour_losses)
            loss = 0.0
            if self.loss_combination == 'uncertainty':
                self.sigma = self.sigma.cuda()
                for i, k in enumerate(losses.keys()):
                    loss_k = losses[k].cuda()
                    if k in ['loss_sem_seg', 'loss_hed_bce']:
                        loss += (torch.exp(-self.sigma[i])
                                 * loss_k + 0.5*self.sigma[i])
                    elif k in ['loss_hed_huber']:
                        loss += 0.5 * \
                            (torch.exp(-self.sigma[i])
                             * loss_k + self.sigma[i])
                    else:
                        raise ValueError('Unkown loss term {}'.format(k))
            else:
                for k, v in losses.items():
                    loss += v

            losses.update({'total_loss': loss})
            losses.update({'weight_sem_seg': self.sigma[0],
                           'weight_hed_huber': self.sigma[1],
                           'weight_hed_bce': self.sigma[2]})
            return losses

        processed_results = []
        for sem_seg_result, contour_result, input_per_image, image_size in zip(
            sem_seg_results, contour_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_r = sem_seg_postprocess(
                sem_seg_result, image_size, height, width)
            contour_r = contour_postprocess(sem_seg_r, contour_result,
                                            height, width,
                                            self.contour_classes)

            processed_results.append(
                {"sem_seg": sem_seg_r, "instances": contour_r})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    contour_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)
    current_segment_id = 0
    segments_info = []
    print(instance_results)
    instance_masks = instance_results.pred_masks.to(
        dtype=torch.bool, device=panoptic_seg.device)
    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in range(len(instance_results)):
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "instance_id": inst_id,
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_limit:
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
