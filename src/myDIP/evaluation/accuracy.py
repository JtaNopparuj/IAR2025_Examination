import numpy as np

def accuracy(output_img, gt_img):

    match_pixel = np.sum(gt_img == output_img)

    height, width = output_img.shape
    total_pixel = height*width
    
    acc = match_pixel/total_pixel

    return acc