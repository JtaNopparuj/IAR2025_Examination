import numpy as np
from myDIP.general import adjustRange
# from myDIP import adjustRange

def gammaCorrection(input_array, gamma, c=1):

    norm_array = adjustRange(input_array, (0,255), (0,1))

    trans_array = c * (norm_array**gamma)

    gamma_trans_img = adjustRange(trans_array, (0,1), (0,255))
    gamma_trans_img = gamma_trans_img.astype(np.uint8)
    
    return gamma_trans_img