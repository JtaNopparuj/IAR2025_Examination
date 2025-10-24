import numpy as np
from sklearn.metrics import confusion_matrix


def confusionMatrix(output_img, gt_img):

    matrix = confusion_matrix(gt_img.flatten(), output_img.flatten()).T
    # print(matrix.shape)

    conf_mat = np.zeros(tuple(i+1 for i in matrix.shape))
    # print(conf_mat.shape)

    conf_mat[:-1, :-1] = matrix

    # ### -> Precision
    # conf_mat[:-1, -1] = np.diagonal(matrix) / np.sum(matrix, axis=1)
    # ### -> Recall 
    # conf_mat[-1, :-1] = np.diagonal(matrix) / np.sum(matrix, axis=0)

    for i in range(matrix.shape[0]):
        ### -> Precision
        conf_mat[i, -1] = matrix[i,i] / np.sum(matrix[i,:]) \
                          if np.sum(matrix[i,:]) !=0 else 0
        ### -> Recall 
        conf_mat[-1, i] = matrix[i,i] / np.sum(matrix[:,i]) \
                          if np.sum(matrix[:,i]) !=0 else 0

    ### -> Accuracy
    conf_mat[-1,-1] = np.sum(np.diagonal(matrix)) / np.sum(matrix)
    # print(matrix)
    # print(conf_mat)

    return conf_mat

def overallConfusionMatrix(output_img_list, gt_img_list):
    
    num_classes = max(len(np.unique(output_img_list)), len(np.unique(gt_img_list)))
    overall_matrix = np.zeros((num_classes+1, num_classes+1))
    

    for output_img, gt_img in zip(output_img_list, gt_img_list):

        matrix = confusion_matrix(gt_img.flatten(), output_img.flatten()).T

        overall_matrix[:-1, :-1] += matrix

    for i in range(num_classes):
        ### -> Precision
        overall_matrix[i, -1] = overall_matrix[i,i] / np.sum(overall_matrix[i,:]) \
                                if np.sum(overall_matrix[i,:]) !=0 else 0
        ### -> Recall 
        overall_matrix[-1, i] = overall_matrix[i,i] / np.sum(overall_matrix[:,i]) \
                                if np.sum(overall_matrix[:,i]) !=0 else 0
        
    ### -> Accuracy
    overall_matrix[-1,-1] = np.sum(np.diagonal(overall_matrix)) / np.sum(overall_matrix[:-1, :-1])

    return overall_matrix
