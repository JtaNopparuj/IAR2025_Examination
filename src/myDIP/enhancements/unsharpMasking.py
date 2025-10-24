from myDIP.filters.smoothing import gaussianFilter
import cv2 as cv
import numpy as np



def unsharpMasking(input_img, blur_filter_size, k=1):

    input_img = input_img.astype(np.float32)

    gauss_filter = gaussianFilter(blur_filter_size)
    low_img = cv.filter2D(input_img,-1,gauss_filter)

    high_img = input_img - low_img

    output_img = input_img + (k*high_img)

    output_img = np.clip(output_img, 0 ,255).astype(np.uint8)

    return output_img
