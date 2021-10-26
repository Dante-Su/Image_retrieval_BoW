import argparse as ap
import cv2
from sklearn.externals import joblib
from scipy.cluster.vq import *
from pylab import *
from PIL import Image
import time
import os



def get_good_match(des1, des2):
    bf = cv2.FlannBasedMatcher_create()
    maches = bf.match(des1, des2)
    maches = sorted(maches, key=lambda x: x.distance)
    good_kp = []
    for j in range(len(maches)):
        if maches[j].distance < 200:
            good_kp.append(maches[j])
    return good_kp


# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required="True")
args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]

# print(os.path.basename(image_path))


since = time.time()

# Load the classifier, class names, scaler, number of clusters and vocabulary
image_paths, voc,voc_node, mo_list,tree_node,tree_leaf ,inverted_file_index= joblib.load("bag-of-words_final_simple_tree.pkl")

# Create feature extraction and keypoint detector objects
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")
fea_det = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)

im_size = im.shape
# print str(im.shape)
im = cv2.resize(im, (int(im_size[1] / 4), int(im_size[0] / 4)))

# kpts = fea_det.detect(im)
# kpts, des = des_ext.compute(im, kpts)

kpts, des = fea_det.detectAndCompute(im, None)
# print(kpts)
# rootsift
# rs = RootSIFT()
# des = rs.compute(kpts, des)

des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
# print(descriptors.shape)
#
test_features = np.zeros((1, len(image_paths)))
words, distance = vq(descriptors, voc_node)
# print(words.shape)
# print(len(distance))
# temp=[]
for i,w in enumerate(words):
    temp=tree_node[w]
    voc_temp=[]
    for index in temp:
        voc_temp.append(voc[index])
    voc_temp=np.array(voc_temp).reshape(-1,len(descriptors[i]))
    #print(voc_temp.shape)
    #print(descriptors[i].shape)

    index,dict=vq(descriptors[i].reshape(1,-1),voc_temp)
    temp=inverted_file_index[tree_node[w][index[0]]]
    if temp[0] == 0:
        print(temp[0])
        continue
    else:
        for ii in temp:
            test_features[0][int(ii) - 1] += 1 / mo_list[int(ii) - 1]


# Perform Tf-Idf vectorization and L2 normalization
# test_features = test_features*idf
# test_features = preprocessing.normalize(test_features, norm='l2')
#
# score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-(test_features[0]))



# print(rank_ID)
# Visualize the results
figure()
gray()
subplot(5, 4, 1)
imshow(im[:, :, ::-1])
axis('off')

ii = 0

# use Spatial Verification
spatial_score = []
for i, ID in enumerate(rank_ID[0:40]):
    # print(str(image_paths[ID]))
    im_1 = cv2.imread(image_paths[ID])
    kpts_1, des_1 = fea_det.detectAndCompute(im_1, None)
    good_kp = get_good_match(des, des_1)
    good_2 = np.expand_dims(good_kp, 1)
    matching = cv2.drawMatchesKnn(im, kpts, im_1, kpts_1, good_2[:20], None, flags=2)
    # plt.figure()
    # plt.imshow('img3.jpg', matching)
    # print("-"*10)
    if len(good_kp) > 10:
        # print("hi")
        ptsA = np.float32([kpts[m.queryIdx].pt for m in good_kp]).reshape(-1, 1, 2)
        ptsB = np.float32([kpts_1[m.trainIdx].pt for m in good_kp]).reshape(-1, 1, 2)
        # cv2.imshow('{}.jpg'.format(str(ID)), matching)
        ransacReprojThreshold = 4

        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)

        spatial_score.append(status.sum())
    else:
        spatial_score.append(2)
spatial_score = np.argsort(-np.array(spatial_score))

for index in spatial_score:
    # print(index)

    # print(status.sum())
    # print(status.shape[0])
    # print(status.sum()/status.shape[0])
    # if (np.float(status.sum()) / np.float(status.shape[0]) ) < 0.15:
    # 	#print("deal")
    # 	continue
    # else:
    if ii > 15:
        break
    else:

        # print("come in")
        img = Image.open(image_paths[rank_ID[index]])
        gray()
        subplot(5, 4, ii + 5)
        ii += 1
        imshow(img)
        axis('off')
end = time.time()
# print(end-since)
plt.savefig('result_img/retrieval_'+os.path.basename(image_path))
# plt.show()