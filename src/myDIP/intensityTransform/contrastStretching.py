import numpy as np


# def contrastStretching(input_array, input_range, output_range):
#     '''
#         Contrast Stretching
#         - input_range:  (input_min, input_max)
#         - output_range: (output_min, output_max)
#     '''
#     r1 ,r2 = input_range
#     s1 ,s2 = output_range

#     input_array = input_array.astype(np.float32)              # - convert to float
#     output_img = s1 + ((input_array-r1)*((s2-s1)/(r2-r1)))  # - contrast stretching
#     print(output_img.min(), output_img.max())
#     output_img = np.clip(output_img, 0, 255)              # - clip value to range [0,255]
#     output_img = output_img.astype(np.uint8)              # - convert to uint8
    
#     return output_img


def contrastStretching(input_array, input_range, output_range):
    '''
        Contrast Stretching
        - input_range:  (input_min, input_max)
        - output_range: (output_min, output_max)
    '''
    r1 ,r2 = input_range
    s1 ,s2 = output_range

    input_array = input_array.astype(np.float32)              # - convert to float
    output_array = np.zeros_like(input_array, np.float32)

    for y in range(input_array.shape[0]):
        for x in range(input_array.shape[1]):

            intensity = input_array[y,x]

            if intensity >= 0 and intensity < r1:
                output_array[y,x] = (s1/r1)*intensity
            elif intensity >= r1 and intensity <= r2:
                output_array[y,x] = ((s2-s1)/(r2-r1))*(intensity-r1) + s1
            else:
                output_array[y,x] = ((255-s2)/(255-r2))*(intensity-r2) + s2

    print(output_array.min(), output_array.max())

    output_array = output_array.astype(np.uint8)              # - convert to uint8
    
    return output_array