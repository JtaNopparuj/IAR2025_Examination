import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
import skimage.morphology as skmorph


# def reconstruction(seed_img, mask_img, method, iteration=5, stre_size=3):

#     init_img = seed_img.copy()

#     stre = skmorph.disk(stre_size)

#     for i in range(iteration):

#         if method == "Dilation":
            
#             dilate_img = cv.dilate(init_img, stre)
#             temp_img = np.minimum(dilate_img, mask_img)

#             if np.array_equal(temp_img, init_img):
#                 break
#             else:
#                 init_img = temp_img
#             # init_img = temp_img
#             # init_img = np.logical_and(dilate_img, mask_img).astype(np.uint8)

#         elif method == "Erosion":
#             erode_img = cv.erode(init_img, stre)
#             init_img = np.maximum(dilate_img, mask_img)
    
#     output_img = init_img
#     print(i)

#     return output_img

def reconstruction(seed_img, mask_img, method, stre_size=3):
    """Morphological reconstruction (Dilation or Erosion) matching skimage behavior."""
    init_img = seed_img.copy()
    stre = skmorph.disk(stre_size)

    # _, label_img = cv.connectedComponents(mask_img.astype(np.uint8))
    # uniq = np.unique(label_img)
    # label_list = np.unique(label_img[seed_img!=0])

    # label_mask_img = np.isin(label_img, label_list)
    # plt.figure()
    # plt.imshow(label_mask_img)
    count = 0
    while True:
        previous = init_img.copy()

        if method == "Dilation":
            dilate_img = cv.dilate(init_img, stre)
            init_img = np.minimum(dilate_img, mask_img)

        elif method == "Erosion":
            erode_img = cv.erode(init_img, stre)
            init_img = np.maximum(erode_img, mask_img)
        else:
            raise ValueError("Invalid method. Use 'Dilation' or 'Erosion'.")

        # Stop if converged (no changes)
        if np.array_equal(previous, init_img):
            break
    
        if count == 1 or count == 3 or count == 8:
            plt.figure()
            plt.imshow(init_img, cmap='gray')
        count += 1
    print(count)
    return init_img