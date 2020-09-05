import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_roi_heads, roi_heads
from detectron2.modeling.backbone import FPN, build_backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage

from torch import nn
import numpy as np

from .postprocessing import contour_postprocess
from .utils import (build_contour_head, build_fpn, build_fpn_block,
                    build_hed_head, build_sem_seg_head,
                    build_semantic_instance_head, get_gt_contours)
from contour.data.builtin import MetadataCatalog


import cv2

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
        # Input
        self.input_format = cfg.INPUT.FORMAT
        self.vis_period = cfg.VIS_PERIOD
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.register_buffer("pixel_mean", torch.Tensor(
            cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(
            cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # Network
        self.backbone = build_backbone(cfg)
        self.fpn = build_fpn(cfg, self.backbone.output_shape())
        self.arch = cfg.MODEL.CONTOUR_NET.ARCH
        self.loss_combination = cfg.MODEL.CONTOUR_NET.LOSS_COMBINATION
        self._build_contour_net(cfg)

        # Post Process
        self.combine_on = cfg.MODEL.CONTOUR_NET.COMBINE.ENABLED
        self.combine_overlap_threshold = \
            cfg.MODEL.CONTOUR_NET.COMBINE.OVERLAP_THRESH
        self.combine_stuff_area_limit = \
            cfg.MODEL.CONTOUR_NET.COMBINE.STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = \
            cfg.MODEL.CONTOUR_NET.COMBINE.INSTANCES_CONFIDENCE_THRESH

    def _build_contour_net(self, cfg):
        if self.arch == 'dual_decoders':
            self.fpn_block = build_fpn_block(cfg, self.fpn.output_shape())
            self.sem_seg_head = build_sem_seg_head(
                cfg, self.fpn_block.output_shape())
            self.hed_head = build_hed_head(cfg, self.backbone.output_shape())
            self.contour_classes = cfg.MODEL.HED_HEAD.NUM_CLASSES
            n_losses = 3 if cfg.MODEL.HED_HEAD.HUBER_ACTIVE else 2
            self._build_loss_combination(n_losses)

        elif self.arch == "dual_blocks":
            self.fpn_block_seg = build_fpn_block(cfg, self.fpn.output_shape())
            self.fpn_block_contour = build_fpn_block(
                cfg, self.fpn.output_shape())
            self.sem_seg_head = build_sem_seg_head(
                cfg, self.fpn_block_seg.output_shape())
            self.contour_head = build_contour_head(
                cfg, self.fpn_block_contour.output_shape())
            self.contour_classes = cfg.MODEL.CONTOUR_HEAD.NUM_CLASSES
            n_losses = 3 if cfg.MODEL.CONTOUR_HEAD.HUBER_ACTIVE else 2
            self._build_loss_combination(n_losses)
        elif self.arch == "dual_heads":
            self.fpn_block = build_fpn_block(cfg, self.fpn.output_shape())
            self.sem_seg_head = build_sem_seg_head(
                cfg, self.fpn_block.output_shape())
            self.contour_head = build_contour_head(
                cfg, self.fpn_block.output_shape())
            self.contour_classes = cfg.MODEL.CONTOUR_HEAD.NUM_CLASSES
            n_losses = 3 if cfg.MODEL.CONTOUR_HEAD.HUBER_ACTIVE else 2
            self._build_loss_combination(n_losses)
        elif self.arch == "single_head":
            self.fpn_block = build_fpn_block(cfg, self.fpn.output_shape())
            self.semantic_instance_head = build_semantic_instance_head(
                cfg, self.fpn_block.output_shape())
            num_classes = cfg.MODEL.SEMANTIC_INSTANCE_HEAD.NUM_CLASSES
            self.contour_classes = num_classes - 19
            n_losses = 2 if cfg.MODEL.SEMANTIC_INSTANCE_HEAD.DUAL_LOSS else 1
            self._build_loss_combination(n_losses)
        else:
            raise ValueError('Unkown architecture for Contour Net.')

    def _build_loss_combination(self, n_losses):
        if self.loss_combination == 'uncertainty':
            self.sigma = nn.Parameter(torch.ones(n_losses))

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
        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, self.backbone.size_divisibility,
                255).tensor
        else:
            gt_sem_seg = None

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
            gt_contours_raw = get_gt_contours(gt_instances, images.image_sizes,
                                          self.contour_classes)
            gt_contours = ImageList.from_tensors(
                gt_contours_raw, self.backbone.size_divisibility).tensor
            gt_contours = gt_contours.to(self.device)
        else:
            gt_instances, gt_contours = None, None

        features = self.backbone(images.tensor)
        (sem_seg_results, sem_seg_losses, contour_results,
         contour_losses) = \
            self._forward_contour_net(features, gt_sem_seg, gt_contours)

        if self.training:
            losses = self.do_train(sem_seg_losses, contour_losses)
            if self.vis_period > 0:
                storage = get_event_storage()
                if (storage.iter % self.vis_period == 0) and storage.iter >= 5000:
                    results = self.inference(batched_inputs, images,
                                             sem_seg_results, contour_results)
                    self.visualize_results(batched_inputs, results, gt_contours_raw)
            return losses

        else:
            return self.inference(batched_inputs, images,
                                     sem_seg_results, contour_results)

    def do_train(self, sem_seg_losses, contour_losses):
        losses = {}
        losses.update(sem_seg_losses)
        if contour_losses is not None:
            losses.update(contour_losses)
        storage = get_event_storage()
        if (self.loss_combination == 'uncertainty') and (storage.iter >= 500):
            loss = 0.0
            self.sigma = self.sigma.cuda()
            loss_keys = list(losses.keys())
            for i, k in enumerate(loss_keys):
                loss_k = losses[k].cuda()
                if k in ['loss_sem_seg', 'loss_hed_bce', 'loss_contour']:
                    if loss_k == 0.0:
                        continue
                    losses[k] = (torch.exp(-self.sigma[i])
                                 * loss_k + 0.5*self.sigma[i])
                elif k in ['loss_hed_huber']:
                    losses[k] = 0.5 * (torch.exp(-self.sigma[i])
                                       * loss_k + self.sigma[i])
                else:
                    raise ValueError('Unkown loss term {}'.format(k))
        return losses

    def inference(self, batched_inputs, images,
                  sem_seg_results, contour_results):
        processed_results = []
        for sem_seg_result, contour_result, input_per_image, image_size in zip(
                sem_seg_results, contour_results, batched_inputs, images.image_sizes):
            height, width = image_size
            sem_seg_r = sem_seg_postprocess(
                sem_seg_result, image_size, height, width)
            instances, contours = contour_postprocess(
                sem_seg_r, contour_result, image_size, 
                height, width, self.contour_classes)
            processed_results.append({"sem_seg": sem_seg_r,
                                      "instances": instances,
                                      "contours": contours})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    instances,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results

    def _forward_contour_net(self, features, gt_sem_seg, gt_contours):
        if self.arch == 'dual_decoders':
            fpn_features = self.fpn_block(self.fpn(features))
            sem_seg_results, sem_seg_losses = self.sem_seg_head(
                fpn_features, gt_sem_seg)
            contour_results, contour_losses = self.hed_head(
                features, gt_contours)
        elif self.arch == "dual_blocks":
            seg_fpn_features = self.fpn_block_seg(self.fpn(features))
            contour_fpn_features = self.fpn_block_contour(self.fpn(features))
            sem_seg_results, sem_seg_losses = self.sem_seg_head(
                seg_fpn_features, gt_sem_seg)
            contour_results, contour_losses = self.contour_head(
                contour_fpn_features, gt_contours)
        elif self.arch == "dual_heads":
            fpn_features = self.fpn_block(self.fpn(features))
            sem_seg_results, sem_seg_losses = self.sem_seg_head(
                fpn_features, gt_sem_seg)
            contour_results, contour_losses = self.contour_head(
                fpn_features, gt_contours)
        elif self.arch == "single_head":
            fpn_features = self.fpn_block(self.fpn(features))
            results, losses = self.semantic_instance_head(
                fpn_features, gt_sem_seg, gt_contours)
            sem_seg_results = \
                results[:, :19, ...] if results is not None else None
            sem_seg_losses = {"loss_sem_seg": losses.get("loss_sem_seg", 0.0)}
            contour_results = torch.argmax(results, dim=1) \
                if results is not None else None
            contour_results[contour_results < 19] = 0
            contour_results[contour_results >= 19] -= 18
            if losses.get("loss_contour", 0.0) != 0.0:
                contour_losses = {"loss_contour": losses["loss_contour"]}
            else:
                contour_losses = None
        else:
            raise ValueError('Unkown architecture for Contour Net.')

        return sem_seg_results, sem_seg_losses, contour_results, contour_losses

    def visualize_results(self, batched_inputs, results, gt_contours):
        """
        A function used to visualize ground truth images and final network predictions.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from .visualization import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.input_format == "BGR":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = Visualizer(img, None)
        if batched_inputs[image_index]["instances"].has('gt_masks'):
            v_gt = v_gt.overlay_instances(
                masks=batched_inputs[image_index]["instances"].gt_masks)
            anno_img = v_gt.get_image()
        else:
            anno_img = np.zeros_like(img)

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(
            masks=results[image_index]["instances"].pred_masks)
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT Instances; Bottom: Predicted Instances"
        storage.put_image(vis_name, vis_img)
        
        cont_gt = Visualizer(img, self.metadata)
        cont_gt = cont_gt.draw_contours(gt_contours[image_index].cpu())
        cont_gt = cont_gt.get_image()
        cont_pred = Visualizer(img, self.metadata)
        cont_pred = cont_pred.draw_contours(results[image_index]["contours"])
        cont_pred = cont_pred.get_image()
        cont_img = np.vstack((cont_gt, cont_pred))
        cont_img = cont_img.transpose(2, 0, 1)
        cont_name = f"Top: GT Contours; Bottom: Predicted Contours"
        storage.put_image(cont_name, cont_img)
        # cv2.imshow(vis_name, vis_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


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
