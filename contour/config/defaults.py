# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
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
_C.MODEL.CENTER_REG_HEAD.CONVS_DIM = 128
_C.MODEL.CENTER_REG_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.CENTER_REG_HEAD.CONV_KERNEL_SIZE = 1
_C.MODEL.CENTER_REG_HEAD.NORM = "GN"


# ---------------------------------------------------------------------------- #
# HED Head
# ---------------------------------------------------------------------------- #
_C.MODEL.HED_HEAD = CN()
_C.MODEL.HED_HEAD.NAME = "HEDHead"
_C.MODEL.HED_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.HED_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.HED_HEAD.NUM_CLASSES = 1  # 9  # (8 Stuff + 1 background)
_C.MODEL.HED_HEAD.HUBER_ACTIVE = True
_C.MODEL.HED_HEAD.DICE_ACTIVE = False
_C.MODEL.HED_HEAD.NORM = ""
_C.MODEL.HED_HEAD.RES5_SUPERVISION = False


# ---------------------------------------------------------------------------- #
# Contour Head
# ---------------------------------------------------------------------------- #
_C.MODEL.CONTOUR_HEAD = CN()
_C.MODEL.CONTOUR_HEAD.NAME = "ContourHead"
_C.MODEL.CONTOUR_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.CONTOUR_HEAD.NUM_CLASSES = 1  # 9  # (8 Stuff + 1 background)
_C.MODEL.CONTOUR_HEAD.HUBER_ACTIVE = True
_C.MODEL.CONTOUR_HEAD.DICE_ACTIVE = False
_C.MODEL.CONTOUR_HEAD.NORM = "GN"


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
_C.MODEL.CONTOUR_NET.ARCH = "dual_heads"
# dual_decoders, dual_blocks, dual_heads, single_head

# options when combining instance & semantic segmentation outputs
_C.MODEL.CONTOUR_NET.COMBINE = CN({"ENABLED": False})
_C.MODEL.CONTOUR_NET.COMBINE.OVERLAP_THRESH = 0.5
_C.MODEL.CONTOUR_NET.COMBINE.STUFF_AREA_LIMIT = 4096
_C.MODEL.CONTOUR_NET.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
_C.MODEL.CONTOUR_NET.LOSS_COMBINATION = 'fixed'  # uncertainty
