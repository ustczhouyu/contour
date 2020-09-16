"""Default config for contour project."""
from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.INPUT.MASK_FORMAT = "bitmask"  # alternative: "bitmask"

# ---------------------------------------------------------------------------- #
# FPN Block
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN_BLOCK = CN()
_C.MODEL.FPN_BLOCK.NAME = "FPNBlock"
_C.MODEL.FPN_BLOCK.IN_FEATURES = ["p2", "p3", "p4", "p5"]

# Number of channels in the 3x3 convs inside FPN heads.
_C.MODEL.FPN_BLOCK.CONVS_DIM = 128
# Outputs from FPN heads are up-scaled to the COMMON_STRIDE stride.
_C.MODEL.FPN_BLOCK.COMMON_STRIDE = 4
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
_C.MODEL.FPN_BLOCK.NORM = "GN"

# ---------------------------------------------------------------------------- #
# HED Decoder
# ---------------------------------------------------------------------------- #
_C.MODEL.HED_DECODER = CN()
_C.MODEL.HED_DECODER.NAME = "HedDecoder"
_C.MODEL.HED_DECODER.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.HED_DECODER.LOSS_WEIGHT = 1.0
_C.MODEL.HED_DECODER.NUM_CLASSES = 1  # 9  # (8 Stuff + 1 background)
_C.MODEL.HED_DECODER.HUBER_ACTIVE = True
_C.MODEL.HED_DECODER.DICE_ACTIVE = False
_C.MODEL.HED_DECODER.NORM = ""
_C.MODEL.HED_DECODER.RES5_SUPERVISION = False

# ---------------------------------------------------------------------------- #
# Semantic Segmentation Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SEM_SEG_HEAD = CN()
_C.MODEL.SEM_SEG_HEAD.NAME = "SemSegHead"
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19  # 19
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255

# ---------------------------------------------------------------------------- #
# Center Regression Head
# ---------------------------------------------------------------------------- #
_C.MODEL.CENTER_REG_HEAD = CN()
_C.MODEL.CENTER_REG_HEAD.NAME = "CenterRegHead"
_C.MODEL.CENTER_REG_HEAD.LOSS_WEIGHT = 0.1

# ---------------------------------------------------------------------------- #
# Contour Head
# ---------------------------------------------------------------------------- #
_C.MODEL.CONTOUR_HEAD = CN()
_C.MODEL.CONTOUR_HEAD.NAME = "ContourHead"
_C.MODEL.CONTOUR_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.CONTOUR_HEAD.NUM_CLASSES = 1  # 9  # (8 Stuff + 1 background)
_C.MODEL.CONTOUR_HEAD.HUBER_ACTIVE = False
_C.MODEL.CONTOUR_HEAD.STEAL_ACTIVE = False

# ---------------------------------------------------------------------------- #
# Semantic with Instance Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SEMANTIC_INSTANCE_HEAD = CN()
_C.MODEL.SEMANTIC_INSTANCE_HEAD.NAME = "SemanticInstanceHead"
_C.MODEL.SEMANTIC_INSTANCE_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.SEMANTIC_INSTANCE_HEAD.NUM_CLASSES = 27  # (19 seg + 1 or 8 contours)
_C.MODEL.SEMANTIC_INSTANCE_HEAD.DUAL_LOSS = False
_C.MODEL.SEMANTIC_INSTANCE_HEAD.HUBER_ACTIVE = True
_C.MODEL.SEMANTIC_INSTANCE_HEAD.DICE_ACTIVE = False
_C.MODEL.SEMANTIC_INSTANCE_HEAD.DUAL_LOSS_WEIGHT = 50.0
_C.MODEL.SEMANTIC_INSTANCE_HEAD.NORM = "GN"
_C.MODEL.SEMANTIC_INSTANCE_HEAD.IGNORE_VALUE = 255

# ---------------------------------------------------------------------------- #
# Contour Net Meta Arch
# ---------------------------------------------------------------------------- #
_C.MODEL.CONTOUR_NET = CN()
# dual_blocks, dual_heads, single_head
_C.MODEL.CONTOUR_NET.ARCH = "seperate_heads"
_C.MODEL.CONTOUR_NET.LOSS_COMBINATION = 'fixed'  # uncertainty

# options when generating the instances from segmentation and contours
_C.MODEL.CONTOUR_NET.POSTPROC = CN()
_C.MODEL.CONTOUR_NET.POSTPROC.CONTOUR_CONF_THRESH = 0.5
_C.MODEL.CONTOUR_NET.POSTPROC.MIN_PIX_AREA = 50  # scaled with postproc_in_stride
_C.MODEL.CONTOUR_NET.POSTPROC.OUTPUT_HEIGHT = 1024
_C.MODEL.CONTOUR_NET.POSTPROC.OUTPUT_WIDTH = 2048

# options when refining the instances using center regression
_C.MODEL.CONTOUR_NET.REFINE = CN({"ENABLED": True})
_C.MODEL.CONTOUR_NET.REFINE.SPLIT_EPS = 8
_C.MODEL.CONTOUR_NET.REFINE.SPLIT_MIN_SAMPLES = 20
_C.MODEL.CONTOUR_NET.REFINE.SPLIT_SAMPLE_SIZE_RATIO = 0.25
_C.MODEL.CONTOUR_NET.REFINE.SPLIT_SAMPLE_MAX_SIZE = 20000
_C.MODEL.CONTOUR_NET.REFINE.MERGE_EPS = 10
_C.MODEL.CONTOUR_NET.REFINE.MERGE_MIN_SAMPLES = 1

# options when combining instance & semantic segmentation outputs
_C.MODEL.CONTOUR_NET.COMBINE = CN({"ENABLED": True})
_C.MODEL.CONTOUR_NET.COMBINE.OVERLAP_THRESH = 0.5
_C.MODEL.CONTOUR_NET.COMBINE.STUFF_AREA_LIMIT = 4096
_C.MODEL.CONTOUR_NET.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5


def get_cfg():
    """Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C.clone()
