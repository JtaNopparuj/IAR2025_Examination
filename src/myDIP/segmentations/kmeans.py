import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def kmeans(input_img , k):

    y, x, c = input_img.shape
    input_rec = input_img.reshape(y*x, c)
    input_rec = input_rec.astype(np.float32)

    # -> K-means clustering
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 0.2)
    _, label_rec, centers = cv.kmeans(input_rec, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # print(centers)
    # print(label_rec.shape)
    # plt.figure()
    # plt.imshow(label_rec.reshape((y,x)))
    centers = centers.astype(np.uint8)
    output_rec = centers[label_rec.flatten()]

    output_img = output_rec.reshape((y,x,c))

    return output_img, centers