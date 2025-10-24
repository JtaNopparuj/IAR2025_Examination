import numpy as np


def iou(output_img, gt_img):

    intersect = np.logical_and(output_img, gt_img)

    union = np.logical_or(output_img, gt_img)

    iou = np.sum(intersect)/np.sum(union)

    return iou