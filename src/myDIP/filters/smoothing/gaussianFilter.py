import cv2 as cv


def gaussianFilter(filter_size):

    gauss_1D = cv.getGaussianKernel(filter_size, -1)
    gauss_filter = gauss_1D * gauss_1D.T

    return gauss_filter

