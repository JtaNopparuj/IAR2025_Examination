import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm

import pyfeats
from scipy.stats import moment
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def fillHoles(input_img):
    '''
        Fill Holes of a Binary Image
    '''
    # -> Create Buffer image
    buffer_img = np.zeros((input_img.shape[0]+2, input_img.shape[1]+2), np.uint8)
    buffer_img[1:-1, 1:-1] = input_img
    # -> Empty image
    empty_img = np.zeros((buffer_img.shape[0]+2, buffer_img.shape[1]+2), np.uint8)
    
    # -> Flood Fill
    _, flood_img, _, _ = cv.floodFill(buffer_img, empty_img, (0, 0), 1)
    flood_img = flood_img[1:-1, 1:-1]
    
    # -> Holes Masking
    hole_img = np.logical_not(flood_img)
    
    # -> Fill Holes
    output_img = np.logical_or(input_img, hole_img) 
    output_img = output_img.astype(np.uint8)

    return output_img

def getLargestConnectedComponent(binary_img):
    '''
        Remove Fragments in the Binary Image
    '''
    binary_img = binary_img.astype(np.uint8)
    # -> Connected Components
    _, label_img = cv.connectedComponents(binary_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    counts = counts[labels!=0]
    labels = labels[labels!=0]

    largest_group_label = labels[np.argmax(counts)]

    output_img = np.zeros_like(binary_img, np.uint8)
    output_img[label_img==largest_group_label] = 1

    return output_img

def teaLeafSegmentation(gray_img):

    seg_img = cv.threshold(gray_img, 10, 255, cv.THRESH_BINARY)[1] 
    seg_img = getLargestConnectedComponent(seg_img)
    seg_img = fillHoles(seg_img)
    return seg_img

def averageIntensity(input_img, mask):

    intensity_list = input_img[mask!=0]
    avg_intensity = np.mean(intensity_list, axis=0)

    return avg_intensity

if __name__ == "__main__":

    tea_leaf_dataset_dir = "../Datasets/Tea_Leaf_Classification/Image/"
    tea_leaf_classes = os.listdir(tea_leaf_dataset_dir)

    X = []
    y = []
    image_name_list = []

    for class_label, tea_leaf_class in enumerate(tqdm(tea_leaf_classes)):
        tea_leaf_dir = os.path.join(tea_leaf_dataset_dir, tea_leaf_class)
        tea_leaf_path_list = sorted(glob(tea_leaf_dir + '/*'))

        for i, path in enumerate(tqdm(tea_leaf_path_list)):

            img_name = os.path.basename(path)
            image_name_list.append(img_name)

            tea_leaf_img = cv.imread(path)
            tea_leaf_img_gray = cv.cvtColor(tea_leaf_img, cv.COLOR_BGR2GRAY)
            tea_leaf_img_hsv = cv.cvtColor(tea_leaf_img, cv.COLOR_BGR2HSV)

            ### -> Segmentation 
            seg_img = teaLeafSegmentation(tea_leaf_img_gray)

            ### -> Feature Extraction
            avg_intensity = averageIntensity(tea_leaf_img_hsv, seg_img)
            lte_feature, lte_labels = pyfeats.lte_measures(tea_leaf_img_gray, mask=seg_img, l=5)

            f_vect = np.concatenate((avg_intensity, lte_feature))

            X.append(f_vect)
            y.append(class_label)

        
    ### -> Normalize features
    X_norm = normalize(X, norm="l2", axis=0)

    X_train, X_test, y_train, y_true, _, test_name_list = train_test_split(X_norm, y, image_name_list, stratify=y, test_size=0.3, random_state=5)
   
    ### -> Classifier
    RF = RandomForestClassifier(n_estimators=500, random_state=5)
    RF.fit(X_train, y_train)

    y_pred = RF.predict(X_test)

    ### -> Evaluation
    print(classification_report(y_true, y_pred, target_names=tea_leaf_classes, digits=3))

    conf_mat = confusion_matrix(y_true, y_pred, labels=[1,0,2,3])
    print(conf_mat)

    # false_predict_name = np.array(test_name_list)[y_pred != y_true]
    # print(false_predict_name)
    # true_label = np.array(y_true)[y_pred != y_true]
    # print(true_label)
    # predict_label = np.array(y_pred)[y_pred != y_true]
    # print(predict_label)