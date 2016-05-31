#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils 
import numpy as np

import os
from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation

from sklearn import svm, grid_search, metrics
from sklearn.externals.joblib import Parallel, delayed

def run_gridSearch(classifier, args, data_training, label_training, data_validation, label_validation):
    from sklearn import metrics
    return metrics.accuracy_score(classifier(**args).fit(data_training, label_training).predict(data_validation), label_validation)

if __name__ == '__main__':
    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to dataset", required="True")
    args = vars(parser.parse_args())
    
    # Get the training classes names and store them in a list
    train_path = args["dataset"]
    training_names = os.listdir(train_path)
    
    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1
    
    # Create feature extraction and keypoint detector objects
    #sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    
    # List where all the descriptors are stored
    des_list = []
    
    for image_path in image_paths:
        im = cv2.imread(image_path)
        #kpts, des = sift.detectAndCompute(im, None)
        kpts, des = surf.detectAndCompute(im, None)
        des_list.append((image_path, des))   
        
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  
    
    # Perform k-means clustering
    k = 100
    voc, variance = kmeans(descriptors, k, 1) 
        
    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
    
    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
    
    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    
    print im_features
    print im_features.shape
    print np.array(image_classes)
    print (np.array(image_classes)).shape
    print "----------------------------------------------"
    # Train the Linear SVM
    #clf = LinearSVC()
    #clf = SVC(C = 100, kernel='rbf')
    #clf.fit(im_features, np.array(image_classes))
    
    
    #clf = SVC(C = 100, kernel='rbf')
    #scores = cross_validation.cross_val_score(clf, im_features, np.array(image_classes), cv=4)
    #print scores
    # output: 0.71428571  0.67857143  0.45833333  0.5  
    
    """
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(im_features, np.array(image_classes), test_size=0.26, random_state=0)
    clf = SVC(C = 100, kernel='rbf').fit(X_train, y_train)
    scores = clf.score(X_test, y_test) 
    print scores
    """
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(im_features, np.array(image_classes), test_size=0.26, random_state=0)
    gamma_range = np.power (10., np.arange (-5, 5, 0.5));
    C_range = np.power (10., np.arange (-5, 5));
    grid_search_params_SVC = \
      [{'kernel' : ['rbf'], 'C' : C_range, 'gamma' : gamma_range},\
       {'kernel' : ['linear'], 'C' : C_range}];
    
    classifier = svm.SVC
    
    grid_search_ans = Parallel(n_jobs = -1)(delayed(run_gridSearch)(classifier, args, X_train, y_train, X_test, y_test) for args in list(grid_search.ParameterGrid(grid_search_params)))
    
    best_params = list(grid_search.ParameterGrid(grid_search_params))[grid_search_ans.index(max(grid_search_ans))]
    
    clf = classifier(**best_params).fit(X_train, y_train)
    
    pred = clf.predict(X_test)
    
    print metrics.classification_report (pred, y_test)
    
    print 'accuracy: ', metrics.accuracy_score(pred, y_test)
    
    
    # Save the SVM
    #joblib.dump((clf, training_names, stdSlr, k, voc), "surf_fm_trained.pkl", compress=3)    
    
    
