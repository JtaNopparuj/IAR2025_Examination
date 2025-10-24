import numpy as np
import cv2 as cv
from myDIP.filters.edge import laplacianFilter


def laplacianSharpening(input_img, center="negative", neighbors=4):

    input_img = input_img.astype(np.float32)

    lpc_filter = laplacianFilter(center, neighbors)
    edge_img = cv.filter2D(input_img, -1, lpc_filter)

    if center == "negative":
        output_img = input_img + ((-1)*edge_img)
    elif center == "positive":
        output_img = input_img + edge_img

    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img, edge_img