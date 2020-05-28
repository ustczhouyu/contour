import numpy as np
import cv2


def rgb_from_pred_contours(score_fuse_feats, conf_thresh=0.5, thinned=False):
    """ Generate rgb visuals from contour_predictions
    score_fuse_feats = torch.sigmoid(score_fuse_feats[0].squeeze()).cpu().numpy()
    Arguments:
        score_fuse_feats {[type]} -- Sigmoid features [cxhxw]
    """
    num_classes = score_fuse_feats.shape[0]
    h, w = score_fuse_feats.shape[1], score_fuse_feats.shape[2]
    r = np.zeros((h, w))
    g = np.zeros((h, w))
    b = np.zeros((h, w))
    rgb = np.zeros((h, w, 3))
    for idx_cls in range(num_classes):
        score_pred = score_fuse_feats[idx_cls, ...]
        score_pred_flag = (score_pred > conf_thresh).astype(np.uint8)
        if thinned:
            score_pred_flag = get_thin_contours(score_pred_flag)
        r[score_pred_flag == 1] = color_dict[idx_cls+11][0]
        g[score_pred_flag == 1] = color_dict[idx_cls+11][1]
        b[score_pred_flag == 1] = color_dict[idx_cls+11][2]
    r[r == 0] = 255
    g[g == 0] = 255
    b[b == 0] = 255
    rgb[:, :, 0] = (r/255.0)
    rgb[:, :, 1] = (g/255.0)
    rgb[:, :, 2] = (b/255.0)

    return rgb


def get_thin_contours(binary_map):
    h, w = binary_map.shape[0], binary_map.shape[1]
    img = np.zeros((h, w), dtype=np.uint8)
    binary_map[binary_map == 1] = 127

    # morphological operator
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # dilated = cv2.dilate(gray, kernel)
    # Closing Morphological Operator
    closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
    # Opening Morphological Operator
    # opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(closed.copy(),
                               cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # select contour based on area and arc length
    # cnts = [c for c in cnts if (cv2.arcLength(c, True) < 1880.0
    # and cv2.arcLength(c, True) >= 10.0)]
    # cnts = [c for c in cnts if (cv2.arcLength(
    #     c, True) < 1880.0 and cv2.contourArea(c, True) >= 150.0)]
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(cnts) > 0:
        img = cv2.drawContours(img, cnts, -1, 1, 2)

    return img


def rgb_from_gt_contours(gt_data):
    """Generate RGB Visual from gt_contours.

    Arguments:
        gt_data {[type]} -- [cxhxw]
    """
    num_classes = gt_data.shape[0]
    h, w = gt_data.shape[1], gt_data.shape[2]
    r = np.zeros((h, w))
    g = np.zeros((h, w))
    b = np.zeros((h, w))
    rgb = np.zeros((h, w, 3))
    for idx_cls in range(num_classes):
        score_pred_flag = gt_data[idx_cls, ...]
        r[score_pred_flag == 1] = color_dict[idx_cls+11][0]
        g[score_pred_flag == 1] = color_dict[idx_cls+11][1]
        b[score_pred_flag == 1] = color_dict[idx_cls+11][2]
    r[r == 0] = 255
    g[g == 0] = 255
    b[b == 0] = 255
    rgb[:, :, 0] = (r/255.0)
    rgb[:, :, 1] = (g/255.0)
    rgb[:, :, 2] = (b/255.0)
    return rgb


# cityscapes
color_dict = {}
color_dict[0] = [128, 64, 128]
color_dict[1] = [244, 35, 232]
color_dict[2] = [70, 70, 70]
color_dict[3] = [102, 102, 156]
color_dict[4] = [190, 153, 153]
color_dict[5] = [153, 153, 153]
color_dict[6] = [250, 170, 30]
color_dict[7] = [220, 220, 0]
color_dict[8] = [107, 142, 35]
color_dict[9] = [152, 251, 152]
color_dict[10] = [70, 130, 180]
color_dict[11] = [220, 20, 60]
color_dict[12] = [255,  0,  0]
color_dict[13] = [0, 0, 142]
color_dict[14] = [0, 0, 70]
color_dict[15] = [0, 60, 100]
color_dict[16] = [0, 80, 100]
color_dict[17] = [0, 0, 230]
color_dict[18] = [119, 11, 32]
