import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from glob import glob
from skimage.util import random_noise
from myDIP.enhancements import flatFieldCorrection
from skimage.exposure import equalize_adapthist
from myDIP.morphology import fillHoles, removeFragments

def selectLargestComponent(input_img):
    input_img = input_img.astype(np.uint8)
    _, label_img = cv.connectedComponents(input_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)
    # print(labels)
    counts = counts[labels!=0]
    labels = labels[labels!=0]

    # print(counts)
    sorted_indices = np.argsort(counts)[::-1]
    labels = labels[sorted_indices]
    counts = counts[sorted_indices]

    # # - Pass Label/Group
    output_img = np.zeros_like(input_img)
    output_img[label_img==labels[0]] = 1

    return output_img

def getCircleMask(input_img):

    circle_mask = selectLargestComponent(input_img)
    plt.figure()
    plt.imshow(circle_mask, cmap='gray')
    circle_mask = fillHoles(circle_mask)
    plt.figure()
    plt.imshow(circle_mask, cmap='gray')
    return circle_mask

def getHalfCircleMask(input_img):

    input_img = input_img.astype(np.uint8)
    _, label_img = cv.connectedComponents(input_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    counts = counts[labels!=0]
    labels = labels[labels!=0]

    sorted_indices = np.argsort(counts)[::-1]
    labels = labels[sorted_indices]
    counts = counts[sorted_indices]
    
    # - Pass Label/Group
    output_img = np.zeros_like(input_img)
    output_img[(label_img==labels[1])|(label_img==labels[2])] = 1

    output_img = fillHoles(output_img)

    return output_img

def removeLargeComponents(input_img):

    input_img = input_img.astype(np.uint8)
    _, label_img = cv.connectedComponents(input_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    counts = counts[labels!=0]
    labels = labels[labels!=0]

    sorted_indices = np.argsort(counts)[::-1]
    labels = labels[sorted_indices]
    counts = counts[sorted_indices]
    
    # - Pass Label/Group
    output_img = input_img.copy()
    output_img[(label_img==labels[0])|(label_img==labels[1])|(label_img==labels[2])] = 0

    return output_img

def letterSegmentation(hsv_img):

    white_img = cv.inRange(hsv_img, (0, 0, 150), (180, 50, 255))

    circle_mask = getCircleMask(white_img)

    white_img_masking = white_img * circle_mask

    white_img_masking_inv = np.logical_not(white_img_masking)

    half_circle_mask = getHalfCircleMask(white_img_masking_inv)
    
    white_letter = white_img * half_circle_mask

    ### -> Red/Green Letter Segmentation
    redgreen_mask = np.logical_or(white_img_masking_inv, half_circle_mask)

    redgreen_mask = removeLargeComponents(redgreen_mask)

    ### -> Combined All letter
    final_mask = np.logical_or(white_letter, redgreen_mask)
    final_mask = removeFragments(final_mask, 0.00001)
    final_mask = final_mask.astype(np.uint8)*255

    return final_mask


def iou(output_img, gt_img):

    intersect = np.logical_and(output_img, gt_img)

    union = np.logical_or(output_img, gt_img)

    iou = np.sum(intersect)/np.sum(union)

    return iou

if __name__ == "__main__":

    input_dir = r"../Datasets/Zambuk_Segmentation/Image/"
    groundtruth_dir = r"../Datasets/Zambuk_Segmentation/Groundtruth/"

    input_path_list = sorted(glob(input_dir + "*"))
    groundtruth_list = sorted(glob(groundtruth_dir + "*"))

    for i, input_path in enumerate(input_path_list):

        img = cv.imread(input_path)
        gt_img = cv.imread(groundtruth_list[i], 0)
        gt_img[gt_img>=128] = 255
        gt_img[gt_img<128] = 0

        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        ### -> Letter Segmentation
        output_img = letterSegmentation(hsv_img)
        
        IoU = iou(output_img, gt_img)
        print(f"Image: {os.path.basename(input_path)} | IoU = {IoU:.4f}")
        plt.figure()
        plt.imshow(rgb_img)
        plt.figure()
        plt.imshow(output_img, cmap='gray')
        plt.show()