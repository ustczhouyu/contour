import os
import copy

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.cityscapes import load_cityscapes_instances, load_cityscapes_semantic


# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),

}


def register_panoptic_cityscapes(root, meta, split='train'):
    image_dir = "cityscapes/leftImg8bit/{split}".format(split=split)
    gt_dir = "cityscapes/gtFine/{split}".format(split=split)
    image_dir = os.path.join(root, image_dir)
    gt_dir = os.path.join(root, gt_dir)
    panoptic_key = "cityscapes_fine_panoptic_seg_{split}".format(split=split)
    DatasetCatalog.register(panoptic_key,
                            lambda: merge_to_panoptic(
                                load_cityscapes_instances(image_dir, gt_dir,
                                                          from_json=False,
                                                          to_polygons=False),
                                load_cityscapes_semantic(image_dir, gt_dir)))
    MetadataCatalog.get(panoptic_key).set(
        image_dir=image_dir, gt_dir=gt_dir,
        evaluator_type="cityscapes_panoptic", **meta
    )


def merge_to_panoptic(detection_dicts, sem_seg_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and sem_seg_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    results = []
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    assert len(sem_seg_file_to_entry) > 0

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
meta = _get_builtin_metadata("cityscapes")
register_panoptic_cityscapes(_root, meta, split='train')
register_panoptic_cityscapes(_root, meta, split='val')
register_panoptic_cityscapes(_root, meta, split='test')
