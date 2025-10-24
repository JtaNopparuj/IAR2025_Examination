import numpy as np


def logTransform(input_array, c=None, to_uint8=True):
    '''
        Logarithm Intensity transformation
    '''
    input_array = input_array.astype(np.float32)

    if c is None:
        c = 255/(np.log(1+np.max(input_array)))
    
    # input_array = input_array.astype(np.float32)

    log_trans_img = c * np.log(1+input_array)

    if to_uint8:
        return log_trans_img.astype(np.uint8)

    return log_trans_img