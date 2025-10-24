import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def sobelEdgeDetector(input_img, thresh_percent):

    sobel_h_filter = np.array([[ 1, 2, 1],
                               [ 0, 0, 0],
                               [-1,-2,-1]])
    
    sobel_v_filter = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    
    sobel_h_img = cv.filter2D(input_img.astype(np.float32), -1, sobel_h_filter)
    sobel_h_img = abs(sobel_h_img)

    sobel_v_img = cv.filter2D(input_img.astype(np.float32), -1, sobel_v_filter)
    sobel_v_img = abs(sobel_v_img)

    sobel_img = sobel_h_img + sobel_v_img

    edge_img = np.where(sobel_img > sobel_img.max()*thresh_percent, 1, 0)
    edge_img = edge_img.astype(np.uint8)

    plt.figure()
    plt.imshow(sobel_img, cmap='gray')

    return edge_img
