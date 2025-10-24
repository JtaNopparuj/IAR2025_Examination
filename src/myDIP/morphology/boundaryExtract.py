import cv2 as cv
import skimage.morphology as skmorph


def boundExtract(input_img):

    stre = skmorph.disk(1)
    erode_img = cv.erode(input_img, stre)

    output_img = input_img - erode_img

    return output_img