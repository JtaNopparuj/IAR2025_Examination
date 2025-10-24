import cv2 as cv
import numpy as np


def fillHoles(input_img):

    # -> Create Buffer image
    buffer_img = np.zeros((input_img.shape[0]+2, 
                           input_img.shape[1]+2), np.uint8)
    buffer_img[1:-1, 1:-1] = input_img

    # -> Empty image
    empty_img = np.zeros((buffer_img.shape[0]+2, 
                          buffer_img.shape[1]+2), np.uint8)

    # -> Flood Fill
    _, flood_img, _, _ = cv.floodFill(buffer_img.copy(), 
                                      empty_img, (0, 0), 1)
    flood_img = flood_img[1:-1, 1:-1]

    # -> Holes Masking
    hole_img = np.logical_not(flood_img)

    # -> Fill Holes
    output_img = np.logical_or(input_img, hole_img)
    output_img = output_img.astype(np.uint8) 

    return output_img
