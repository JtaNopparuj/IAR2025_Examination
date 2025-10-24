import numpy as np

def adjustRange(input_array, input_range, output_range):
    '''
    
    '''
    input_min, input_max = input_range
    output_min, output_max = output_range

    input_array = input_array.astype(np.float32)

    # -> Convert to range [0,1]
    norm_array = (input_array - input_min)/  \
                 (input_max - input_min)
    
    # -> Convert from range [0,1] to [output_min, output_max]
    output_array = (norm_array * (output_max-output_min)) \
                    + output_min

    return output_array