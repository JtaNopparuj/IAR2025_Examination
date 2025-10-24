import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 



def removeFragments(input_img, thresh_percent=0.1):
    '''
        Remove Fragments in the Binary Image
    '''
    # -> Connected Components
    input_img = input_img.astype(np.uint8)
    _, label_img = cv.connectedComponents(input_img, connectivity=4)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)
    # print(labels)

    ### -> Fragments Searching
    # - Cut-off Value
    count_pixels = input_img.shape[0] * input_img.shape[1]
    count_thresh = int(count_pixels * thresh_percent)
    # - Thresholding
    pass_index = np.argwhere(counts > count_thresh).flatten()
    # print(pass_index)
    
    assert len(pass_index) > 1, "All Objects are removed, Try reducing 'thresh_percent' value"

    # - Pass Label/Group
    output_img = np.isin(label_img, pass_index[1:])
    output_img = output_img.astype(np.uint8)

    return output_img