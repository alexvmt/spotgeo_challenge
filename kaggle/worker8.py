# setup

import matplotlib.pyplot as plt

import numpy as np

from collections import defaultdict

import json

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

import itertools

import random

from skimage import measure

import pickle


def read_image(path):
    return plt.imread(path)

def read_annotation_file(path):
    with open(path) as annotation_file:
        annotation_list = json.load(annotation_file)
    # Transform list of annotations into dictionary
    annotation_dict = {}
    for annotation in annotation_list:
        sequence_id = annotation['sequence_id']
        if sequence_id not in annotation_dict:
            annotation_dict[sequence_id] = {}
        annotation_dict[sequence_id][annotation['frame']] = annotation['object_coords']
    return annotation_dict

random.seed(0)

def random_different_coordinates(coords, size_x, size_y, pad):
    """ Returns a random set of coordinates that is different from the provided coordinates, 
    within the specified bounds.
    The pad parameter avoids coordinates near the bounds."""
    good = False
    while not good:
        good = True
        c1 = random.randint(pad + 1, size_x - (pad + 1))
        c2 = random.randint(pad + 1, size_y -( pad + 1))
        for c in coords:
            if c1 == c[0] and c2 == c[1]:
                good = False
                break
    return (c1,c2)

def extract_neighborhood(x, y, arr, radius):
    """ Returns a 1-d array of the values within a radius of the x,y coordinates given """
    return arr[(x - radius) : (x + radius + 1), (y - radius) : (y + radius + 1)].ravel()

def check_coordinate_validity(x, y, size_x, size_y, pad):
    """ Check if a coordinate is not too close to the image edge """
    return x >= pad and y >= pad and x + pad < size_x and y + pad < size_y

def generate_labeled_data(image_path, annotation, nb_false, radius):
    """ For one frame and one annotation array, returns a list of labels 
    (1 for true object and 0 for false) and the corresponding features as an array.
    nb_false controls the number of false samples
    radius defines the size of the sliding window (e.g. radius of 1 gives a 3x3 window)"""
    features,labels = [],[]
    im_array = read_image(image_path)
    # True samples
    for obj in annotation:
        obj = [int(x + .5) for x in obj] #Project the floating coordinate values onto integer pixel coordinates.
        # For some reason the order of coordinates is inverted in the annotation files
        if check_coordinate_validity(obj[1],obj[0],im_array.shape[0],im_array.shape[1],radius):
            features.append(extract_neighborhood(obj[1],obj[0],im_array,radius))
            labels.append(1)
    # False samples
    for i in range(nb_false):
        c = random_different_coordinates(annotation,im_array.shape[1],im_array.shape[0],radius)
        features.append(extract_neighborhood(c[1],c[0],im_array,radius))
        labels.append(0)
    return np.array(labels),np.stack(features,axis=1)

def generate_labeled_set(annotation_array, path, sequence_id_list, radius, nb_false):
    # Generate labeled data for a list of sequences in a given path
    labels,features = [],[]
    for seq_id in sequence_id_list:
        for frame_id in range(1,6):
            d = generate_labeled_data(f"{path}{seq_id}/{frame_id}.png",
                                    annotation_array[seq_id][frame_id],
                                    nb_false,
                                    radius)
            labels.append(d[0])
            features.append(d[1])
    return np.concatenate(labels,axis=0), np.transpose(np.concatenate(features,axis=1))

def classify_image(im, model, radius):
    n_features=(2*radius+1)**2 #Total number of pixels in the neighborhood
    feat_array=np.zeros((im.shape[0],im.shape[1],n_features))
    for x in range(radius+1,im.shape[0]-(radius+1)):
        for y in range(radius+1,im.shape[1]-(radius+1)):
            feat_array[x,y,:]=extract_neighborhood(x,y,im,radius)
    all_pixels=feat_array.reshape(im.shape[0]*im.shape[1],n_features)
    pred_pixels=model.predict(all_pixels).astype(np.bool_)
    pred_image=pred_pixels.reshape(im.shape[0],im.shape[1])
    return pred_image

def extract_centroids(pred, bg):
    conn_comp=measure.label(pred, background=bg)
    object_dict=defaultdict(list) #Keys are the indices of the connected components and values are arrrays of their pixel coordinates 
    for (x,y),label in np.ndenumerate(conn_comp):
            if label != bg:
                object_dict[label].append([x,y])
    # Mean coordinate vector for each object, except the "0" label which is the background
    centroids={label: np.mean(np.stack(coords),axis=0) for label,coords in object_dict.items()}
    object_sizes={label: len(coords) for label,coords in object_dict.items()}
    return centroids, object_sizes

def filter_large_objects(centroids,object_sizes, max_size):
    small_centroids={}
    for label,coords in centroids.items():
            if object_sizes[label] <= max_size:
                small_centroids[label]=coords
    return small_centroids

def predict_objects(sequence_id, frame_id, model, radius, max_size):
    test_image = plt.imread(f"../input/spotgeo/test/test/{sequence_id}/{frame_id}.png")
    test_pred=classify_image(test_image, model, radius)
    test_centroids, test_sizes = extract_centroids(test_pred, 0)
    test_centroids = filter_large_objects(test_centroids, test_sizes, max_size)
    # Switch x and y coordinates for submission
    if len(test_centroids.values()) > 0:
        sub=np.concatenate([c[np.array([1,0])].reshape((1,2)) for c in test_centroids.values()])
        #np array converted to list for json seralization, truncated to the first 30 elements
        return sub.tolist()[0:30]
    else:
        return []


# load model

# rf
model = pickle.load(open('../input/spotgeo-models/rf.sav', 'rb'))

# xgb
#model = pickle.load(open('../input/spotgeo-models/xgb.sav', 'rb'))

# nn5 (sklearn)
#model = pickle.load(open('../input/spotgeo-models/nn5.sav', 'rb'))

# nn5 (keras)
#from keras.models import load_model

#def f1(y_true, y_pred):
#    def recall(y_true, y_pred):
#        """Recall metric.

#        Only computes a batch-wise average of recall.

#        Computes the recall, a metric for multi-label classification of
#        how many relevant items are selected.
#        """
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#        recall = true_positives / (possible_positives + K.epsilon())
#        return recall

#    def precision(y_true, y_pred):
#        """Precision metric.

#        Only computes a batch-wise average of precision.

#        Computes the precision, a metric for multi-label classification of
#        how many selected items are relevant.
#        """
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#        precision = true_positives / (predicted_positives + K.epsilon())
#        return precision
#    precision = precision(y_true, y_pred)
#    recall = recall(y_true, y_pred)
#    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#model = load_model('../input/spotgeo-models/nn5', custom_objects={'f1': f1})

#def classify_image(im, model, radius):
#    n_features=(2*radius+1)**2 #Total number of pixels in the neighborhood
#    feat_array=np.zeros((im.shape[0],im.shape[1],n_features))
#    for x in range(radius+1,im.shape[0]-(radius+1)):
#        for y in range(radius+1,im.shape[1]-(radius+1)):
#            feat_array[x,y,:]=extract_neighborhood(x,y,im,radius)
#    all_pixels=feat_array.reshape(im.shape[0]*im.shape[1],n_features)
#    pred_pixels=model.predict_classes(all_pixels).astype(np.bool_)
#    pred_image=pred_pixels.reshape(im.shape[0],im.shape[1])
#    return pred_image


# classify sequences

radius = 3
submission = []

for seq_id in range(4487,5120+1):
    
    for frm_id in range(1,6):
        
        sub_list = predict_objects(seq_id,frm_id,model,radius,1)
        
        submission.append({"sequence_id" : seq_id,
                           "frame" : frm_id,
                           "num_objects" : len(sub_list),
                           "object_coords" : sub_list})

with open('/kaggle/working/my_submission_4487_5120.json', 'w') as outfile:
    json.dump(submission, outfile)