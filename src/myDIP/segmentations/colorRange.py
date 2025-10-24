import numpy as np



def euclideanDistance(input_rec, center_pos):
    '''
        Euclidean Distances
    '''
    # -> Find Euclidean Distances
    dist_rec = np.sqrt(np.sum((input_rec - center_pos)**2, axis=1))

    return dist_rec

def mahalanobisDistance(input_rec, center_pos):
    '''
        Mahalanobis Distances
    '''
    ### -> Find Invert Covariance
    cov = np.cov(input_rec.T)
    inv_cov = np.linalg.inv(cov)
    
    ### -> Find Mahalanobis Distance
    delta = input_rec - center_pos
    dist_rec = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))

    return dist_rec

#### -> Color Range Distance Function
def colorRange(input_img, center_pos, r_cutoff, dist_func="Euclidean"):
    '''
        Color Range Segmentation
    '''
    # -> Record Transformation
    y, x, c = input_img.shape
    input_rec = input_img.reshape(y*x, c)
    # -> Center Position Expansion for Element-wise processing
    center_pos = center_pos * np.ones((y*x, 1))

    # -> Distances Calculation
    distanceFunction = {
                        "Euclidean": euclideanDistance(input_rec, center_pos),
                        "Mahalanobis": mahalanobisDistance(input_rec, center_pos)
                       }
    dist_rec = distanceFunction[dist_func]

    # -> Cutoff Color
    output_rec = np.where(dist_rec > r_cutoff, 0, 255)

    # -> Record Invert Transformation
    output_img = output_rec.reshape((y, x))

    return output_img