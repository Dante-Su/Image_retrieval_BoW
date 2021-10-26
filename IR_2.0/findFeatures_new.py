import numpy as np
from PIL import Image
import sklearn
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
#from rootsift import RootSIFT
import math

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
# train_path = "dataset/training/"

training_names = os.listdir(train_path)

numWords = 1000

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path]

# Create feature extraction and keypoint detector objects
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")
fea_det = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

# Calculate the histogram of features


for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    im_size = im.shape

    # print str(im.shape)
    # if im_size[1] > im_size[0]:
    #     im = cv2.resize(im,(imagesize_0,imagesize_1))
    # else:
    #     im = cv2.resize(im,(imagesize_1,imagesize_0))
    # print str(im.shape)

    im = cv2.resize(im, (im_size[1] / 4, im_size[0] / 4))

    print ("Extract SIFT of %s image, %d of %d images" %(training_names[i], i, len(image_paths)))
    # kpts = fea_det.detect(im)
    # kpts, des = des_ext.compute(im, kpts)
    kpts, des = fea_det.detectAndCompute(im, None)
    # rootsift
    # rs = RootSIFT()
    # des = rs.compute(kpts, des)
    des_list.append((image_path, des))
    # print str(des.shape)

# Stack all the descriptors vertically in a numpy array
# downsampling = 2
# descriptors = des_list[0][1][::downsampling,:]
# for image_path, descriptor in des_list[1:]:
#     # print np.size(descriptor)
#     # print image_path
#     # print descriptor
#     descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))


# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    # print np.size(descriptor)
    # print descriptor
    # if np.size(descriptor) != 0:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
print ("Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1)

print('voc', voc.shape)
print('variance', variance)

nodes_num=50
voc_node,variance_node=kmeans(voc,nodes_num,1)

leaf_to_node,dict=vq(voc,voc_node)
# print(leaf_to_node)


# Calculate the histogram of features
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    #print('words',type(words))
    #print('distance',type(distance))
    #print(words.shape)
    #print(distance.shape)
    for w in words:
        im_features[i][w] += 1

mo_list=np.sqrt(np.sum(im_features*im_features,axis=1))


# build the inverted file index
inverted_file_index=np.zeros((numWords,1))
inverted_file_index=inverted_file_index.tolist()
for i in range(len(image_paths)):
    words,distance=vq(des_list[i][1],voc)
    for w in words:
        if inverted_file_index[w][0]==0:
            inverted_file_index[w][0]=str(i+1)
            #print(w)
        else:
            inverted_file_index[w].append(str(i+1))

# use Vocabulary Tree(hierarchical clustering)
tree_node=[]
tree_leaf=[]
for ii in range(len(voc)):
    tree_leaf.append(ii)
for ii in range(nodes_num):
    tree_node.append([])
for ii in range(numWords):
    tree_node[leaf_to_node[ii]].append(tree_leaf[ii])



joblib.dump(( image_paths,voc,voc_node, mo_list,tree_node,tree_leaf,inverted_file_index), "bag-of-words_final_simple_tree.pkl", compress=3)