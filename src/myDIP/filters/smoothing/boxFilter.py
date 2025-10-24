import numpy as np


def boxFilter(filter_size):

    box_filter = np.ones((filter_size, filter_size))
    # box_filter = (1/(filter_size*filter_size))*box_filter
    box_filter /= (filter_size*filter_size)

    return box_filter