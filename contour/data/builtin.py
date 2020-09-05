import os
import copy

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.cityscapes import load_cityscapes_instances, load_cityscapes_semantic

CITYSCAPES_CATEGORIES = [
    {'color': [255, 255, 255], 'isthing': 0, 'id': 0, 'name': 'road'},
    {'color': [244, 35, 232], 'isthing': 0, 'id': 1, 'name': 'sidewalk'},
    {'color': [70, 70, 70], 'isthing': 0, 'id': 2, 'name': 'building'},
    {'color': [102, 102, 156], 'isthing': 0, 'id': 3, 'name': 'wall'},
    {'color': [190, 153, 153], 'isthing': 0, 'id': 4, 'name': 'fence'},
    {'color': [153, 153, 153], 'isthing': 0, 'id': 5, 'name': 'pole'},
    {'color': [250, 170, 30], 'isthing': 0, 'id': 6, 'name': 'traffic light'},
    {'color': [220, 220, 0], 'isthing': 0, 'id': 7, 'name': 'traffic sign'},
    {'color': [107, 142, 35], 'isthing': 0, 'id': 8, 'name': 'vegetation'},
    {'color': [152, 251, 152], 'isthing': 0, 'id': 9, 'name': 'terrain'},
    {'color': [70, 130, 180], 'isthing': 0, 'id': 10, 'name': 'sky'},
    {'color': [220, 20, 60], 'isthing': 1, 'id': 11, 'name': 'person'},
    {'color': [255, 0, 0], 'isthing': 1, 'id': 12, 'name': 'rider'},
    {'color': [0, 0, 142], 'isthing': 1, 'id': 13, 'name': 'car'},
    {'color': [0, 0, 70], 'isthing': 1, 'id': 14, 'name': 'truck'},
    {'color': [0, 60, 100], 'isthing': 1, 'id': 15, 'name': 'bus'},
    {'color': [0, 80, 100], 'isthing': 1, 'id': 16, 'name': 'train'},
    {'color': [0, 0, 230], 'isthing': 1, 'id': 17, 'name': 'motorcycle'},
    {'color': [119, 11, 32], 'isthing': 1, 'id': 18, 'name': 'bicycle'}
]


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


def _get_cityscapes_panoptic_meta(root):
    """
    Returns metadata of the cityscapes panoptic segmentation dataset.
    """
    meta = _get_builtin_metadata("cityscapes")
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"]
                    for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    panoptic_json = 'cityscapes/gtFine/cityscapes_panoptic_val.json'
    panoptic_root = 'cityscapes/gtFine/cityscapes_panoptic_val'
    ret = {"stuff_colors": stuff_colors, 'thing_colors': thing_colors,
           'panoptic_json': os.path.join(root, panoptic_json),
           'panoptic_root': os.path.join(root, panoptic_root)}
    meta.update(ret)
    return meta


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
meta = _get_cityscapes_panoptic_meta(_root)
register_panoptic_cityscapes(_root, meta, split='train')
register_panoptic_cityscapes(_root, meta, split='val')
register_panoptic_cityscapes(_root, meta, split='test')
