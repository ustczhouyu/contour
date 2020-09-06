"""ContourNet model."""
import numpy as np
import torch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from torch import nn

# pylint: disable=relative-beyond-top-level
from ..data.builtin import MetadataCatalog
from .postprocessing import contour_postprocess
from .utils import (build_center_reg_head, build_contour_head, build_fpn,
                    build_fpn_block, build_hed_decoder, build_sem_seg_head,
                    get_gt)
from .visualization import ImageVisualizer as Visualizer

__all__ = ["ContourNet"]


# pylint: disable=too-many-instance-attributes
@META_ARCH_REGISTRY.register()
class ContourNet(nn.Module):
    """
    Meta Architecture for ContourNet that outputs instance or panoptic seg
     using contours.

    Seperate Decoders:
               |-- fpn -- fpn_block -- sem_seg_head -----|
    Backbone --
               |-- hed_decoder --------------------------|-- postproc -- instance

    Seperate Heads:
                                 |--sem_seg_head--|
    Backbone--|--fpn--fpn_block--|--contour_head--|--postproc--instance--|
                                 |                                       |
                                 |--center_reg_head------------|--merged instances
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
        self.sigma = None

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
            self.center_reg_head = build_center_reg_head(
                cfg, self.fpn_block.output_shape())
            self.hed_decoder = build_hed_decoder(
                cfg, self.backbone.output_shape())

        elif self.arch == "dual_blocks":
            self.fpn_block_seg = build_fpn_block(cfg, self.fpn.output_shape())
            self.fpn_block_contour = build_fpn_block(
                cfg, self.fpn.output_shape())
            self.sem_seg_head = build_sem_seg_head(
                cfg, self.fpn_block_seg.output_shape())
            self.contour_head = build_contour_head(
                cfg, self.fpn_block_contour.output_shape())
            self.center_reg_head = build_center_reg_head(
                cfg, self.fpn_block_seg.output_shape())

        elif self.arch == "dual_heads":
            self.fpn_block = build_fpn_block(cfg, self.fpn.output_shape())
            self.sem_seg_head = build_sem_seg_head(
                cfg, self.fpn_block.output_shape())
            self.contour_head = build_contour_head(
                cfg, self.fpn_block.output_shape())
            self.center_reg_head = build_center_reg_head(
                cfg, self.fpn_block.output_shape())
        else:
            raise ValueError('Unkown architecture for Contour Net.')

        self.contour_classes = cfg.MODEL.CONTOUR_HEAD.NUM_CLASSES
        n_losses = 4 if cfg.MODEL.CONTOUR_HEAD.HUBER_ACTIVE else 3
        self._build_loss_combination(n_losses)

    def _build_loss_combination(self, n_losses):
        if self.loss_combination == 'uncertainty':
            # pylint: disable=no-member
            self.sigma = nn.Parameter(torch.ones(n_losses))

    @property
    def device(self):
        """Get device."""
        return self.pixel_mean.device

    # pylint: disable=arguments-differ, too-many-locals
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
                  "height", "width" (int): the output resolution of the model, used in postprocess.
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
            gt_contours_raw, gt_offsets_raw = get_gt(gt_instances,
                                                     images.image_sizes,
                                                     self.contour_classes)
            gt_contours = ImageList.from_tensors(gt_contours_raw,
                                                 self.backbone.size_divisibility).tensor
            gt_contours = gt_contours.to(self.device)
            gt_offsets = ImageList.from_tensors(gt_offsets_raw,
                                                self.backbone.size_divisibility).tensor
            gt_offsets = gt_offsets.to(self.device)
        else:
            gt_instances, gt_contours, gt_offsets = None, None, None

        features = self.backbone(images.tensor)
        (sem_seg_results, sem_seg_losses, contour_results,
         contour_losses, center_reg_results, center_reg_losses) = \
            self._forward_contour_net(
                features, gt_sem_seg, gt_contours, gt_offsets)

        if self.training:
            losses = self.update_losses(
                sem_seg_losses, contour_losses, center_reg_losses)
            if self.vis_period > 0:
                storage = get_event_storage()
                if (storage.iter % self.vis_period == 0) and storage.iter >= 100:
                    results = self.postprocess(images, sem_seg_results, contour_results,
                                               center_reg_results)
                    self.visualize_results(
                        batched_inputs, results, gt_contours_raw, gt_offsets_raw)
            return losses

        return self.postprocess(images, sem_seg_results, contour_results,
                                center_reg_results)

    def update_losses(self, sem_seg_losses, contour_losses, center_reg_losses):
        """Update losses."""
        losses = {}
        losses.update(sem_seg_losses)
        if contour_losses is not None:
            losses.update(contour_losses)
        if center_reg_losses is not None:
            losses.update(center_reg_losses)
        storage = get_event_storage()
        if (self.loss_combination == 'uncertainty') and (storage.iter >= 500):
            self.sigma = self.sigma.cuda()
            loss_keys = list(losses.keys())
            for i, k in enumerate(loss_keys):
                loss_k = losses[k].cuda()
                if k in ['loss_sem_seg', 'loss_hed_bce', 'loss_contour']:
                    if loss_k == 0.0:
                        continue
                    # pylint: disable=no-member
                    losses[k] = (torch.exp(-self.sigma[i])
                                 * loss_k + 0.5*self.sigma[i])
                elif k in ['loss_hed_huber', 'loss_center_reg']:
                    # pylint: disable=no-member
                    losses[k] = 0.5 * (torch.exp(-self.sigma[i])
                                       * loss_k + self.sigma[i])
                else:
                    raise ValueError('Unkown loss term {}'.format(k))
        return losses

    # pylint: disable=too-many-arguments
    def postprocess(self, images, sem_seg_results,
                    contour_results, center_reg_results):
        """Perform postprocess."""
        processed_results = []
        for sem_seg_result, contour_result, center_reg_result, image_size in zip(
                sem_seg_results, contour_results, center_reg_results, images.image_sizes):
            height, width = image_size
            sem_seg_r = sem_seg_postprocess(
                sem_seg_result, image_size, height, width)
            sem_seg, instances, contours, offsets = contour_postprocess(
                sem_seg_r, contour_result, center_reg_result,
                image_size, height, width, self.contour_classes)
            processed_results.append({"sem_seg": sem_seg,
                                      "instances": instances,
                                      "contours": contours,
                                      "offsets": offsets})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    instances,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results

    def _forward_contour_net(self, features, gt_sem_seg, gt_contours, gt_offsets):
        """Forward helper function."""
        if self.arch == 'dual_decoders':
            fpn_features = self.fpn_block(self.fpn(features))
            sem_seg_results, sem_seg_losses = self.sem_seg_head(
                fpn_features, gt_sem_seg)
            center_reg_results, center_reg_losses = self.center_reg_head(
                fpn_features, gt_offsets)
            contour_results, contour_losses = self.hed_decoder(
                features, gt_contours)
        elif self.arch == "dual_blocks":
            seg_fpn_features = self.fpn_block_seg(self.fpn(features))
            contour_fpn_features = self.fpn_block_contour(self.fpn(features))
            sem_seg_results, sem_seg_losses = self.sem_seg_head(
                seg_fpn_features, gt_sem_seg)
            center_reg_results, center_reg_losses = self.center_reg_head(seg_fpn_features,
                                                                         gt_offsets)
            contour_results, contour_losses = self.contour_head(
                contour_fpn_features, gt_contours)
        elif self.arch == "dual_heads":
            fpn_features = self.fpn_block(self.fpn(features))
            sem_seg_results, sem_seg_losses = self.sem_seg_head(
                fpn_features, gt_sem_seg)
            contour_results, contour_losses = self.contour_head(
                fpn_features, gt_contours)
            center_reg_results, center_reg_losses = self.center_reg_head(
                fpn_features, gt_offsets)
        else:
            raise ValueError('Unkown architecture for Contour Net.')

        return (sem_seg_results, sem_seg_losses, contour_results, contour_losses,
                center_reg_results, center_reg_losses)

    def visualize_results(self, batched_inputs, results, gt_contours, gt_offsets):
        """
        A function used to visualize ground truth images and final network predictions.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"].cpu().numpy()
        seg_anno = batched_inputs[image_index]["sem_seg"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.input_format == "BGR":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = Visualizer(img, None)
        if batched_inputs[image_index]["instances"].has('gt_masks'):
            v_gt = v_gt.overlay_instances(
                masks=batched_inputs[image_index]["instances"].gt_masks)
            inst_anno_img = v_gt.get_image()
        else:
            inst_anno_img = np.zeros_like(img)

        seg_anno_img = Visualizer(img, self.metadata)
        seg_anno_img = seg_anno_img.draw_sem_seg(sem_seg=seg_anno)
        seg_anno_img = seg_anno_img.get_image()
        seg_pred = Visualizer(img, self.metadata)
        seg_pred = seg_pred.draw_sem_seg(sem_seg=results[image_index]["sem_seg"])
        seg_pred = seg_pred.get_image()
        seg_vis_img = np.vstack((seg_anno_img, seg_pred))
        seg_vis_img = seg_vis_img.transpose(2, 0, 1)
        seg_vis_name = "Top: GT Sem Seg; Bottom: Predicted Sem Seg"
        storage.put_image(seg_vis_name, seg_vis_img)

        inst_pred = Visualizer(img, self.metadata)
        inst_pred = inst_pred.overlay_instances(
            masks=results[image_index]["instances"].pred_masks)
        inst_pred = inst_pred.get_image()
        inst_vis_img = np.vstack((inst_anno_img, inst_pred))
        inst_vis_img = inst_vis_img.transpose(2, 0, 1)
        inst_vis_name = "Top: GT Instances; Bottom: Predicted Instances"
        storage.put_image(inst_vis_name, inst_vis_img)

        cont_gt = Visualizer(img, self.metadata)
        cont_gt = cont_gt.draw_contours(gt_contours[image_index].cpu())
        cont_gt = cont_gt.get_image()
        cont_pred = Visualizer(img, self.metadata)
        cont_pred = cont_pred.draw_contours(results[image_index]["contours"])
        cont_pred = cont_pred.get_image()
        cont_img = np.vstack((cont_gt, cont_pred))
        cont_img = cont_img.transpose(2, 0, 1)
        cont_name = "Top: GT Contours; Bottom: Predicted Contours"
        storage.put_image(cont_name, cont_img)

        offset_gt = Visualizer(img, None)
        offset_gt = offset_gt.draw_offsets(gt_offsets[image_index].cpu())
        offset_gt = offset_gt.get_image()
        offset_pred = Visualizer(img, None)
        offset_pred = offset_pred.draw_offsets(results[image_index]["offsets"])
        offset_pred = offset_pred.get_image()
        offset_img = np.vstack((offset_gt, offset_pred))
        offset_img = offset_img.transpose(2, 0, 1)
        offset_name = "Top: GT Offsets; Bottom: Predicted Offsets"
        storage.put_image(offset_name, offset_img)


# pylint: disable=too-many-locals
def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
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
    # pylint: disable=no-member
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)
    current_segment_id = 0
    segments_info = []
    # pylint: disable=no-member
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
