import cv2 as cv
import numpy as np
from myDIP.filters.smoothing import gaussianFilter

def flatFieldCorrection(input_img, gauss_size, dark_img=None):

    input_img = input_img.astype(np.float32)

    if dark_img is None:
        dark_img = np.zeros_like(input_img)

    gauss_filter = gaussianFilter(gauss_size)
    ff_img = cv.filter2D(input_img, -1, gauss_filter)

    # ff_mean = ff_img.mean()

    eps = 1e-8
    output_img = (ff_img.mean()/(ff_img-dark_img+eps)) * (input_img - dark_img)

    # print(output_img.min(), output_img.max())

    output_img = np.clip(output_img,0,255).astype(np.uint8)

    return output_img