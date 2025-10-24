import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detector(input_img, filter, thresh_percent):

    input_img = input_img.astype(np.float32)

    res_img = cv.filter2D(input_img, -1, filter)

    abs_res_img = np.abs(res_img)

    thresh_val = abs_res_img.max() * thresh_percent

    output_img = np.where(abs_res_img > thresh_val, 255, 0)
    output_img = output_img.astype(np.uint8)

    plt.figure()
    plt.imshow(res_img, cmap='gray')
    plt.figure()
    plt.imshow(abs_res_img, cmap='gray')

    return output_img
