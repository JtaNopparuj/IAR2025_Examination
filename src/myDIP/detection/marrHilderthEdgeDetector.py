import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from myDIP.filters.edge import laplacianFilter

def zeroCrossing(input_img): 

    thres_img = np.where(input_img > 0, 1, 0)  

    plt.figure()
    plt.imshow(thres_img, cmap='gray')

    zero_cross_img = np.zeros_like(thres_img)

    input_pad = np.pad(thres_img, ((1,1), (1,1)), "constant", constant_values=0)

    neighbor_4 = np.array([[-1,  0],
                           [ 1,  0],
                           [ 0, -1],
                           [ 0,  1]])

    for i in range(1, input_pad.shape[0]-1):
        for j in range(1, input_pad.shape[1]-1):
            
            if input_pad[i, j] == 0:
                neighbor_idx = np.array([i,j]) + neighbor_4
                if input_pad[neighbor_idx[:,0], neighbor_idx[:,1]].sum() > 0:
                    zero_cross_img[i-1, j-1] = 1

    return zero_cross_img

def marrHilderthEdgeDetector(input_img, kernel_size, thres_percent=None):

    blur_img = cv.GaussianBlur(input_img, (kernel_size, kernel_size), 0)

    lpc_filter = laplacianFilter("positive", 8)

    LoG_img = cv.filter2D(blur_img.astype(np.float32), -1, lpc_filter)

    if thres_percent is not None:
        abs_LoG_img = np.abs(LoG_img)
        max_val = np.max(abs_LoG_img)
        LoG_img[abs_LoG_img < (max_val*thres_percent)] = 0

    zero_cross_img = zeroCrossing(LoG_img)

    plt.figure()
    plt.imshow(input_img, cmap='gray', vmin=0, vmax=255)
    plt.figure()
    plt.imshow(blur_img, cmap='gray', vmin=0, vmax=255)
    plt.figure()
    plt.imshow(LoG_img, cmap='gray')
    plt.figure()
    plt.imshow(zero_cross_img, cmap='gray')