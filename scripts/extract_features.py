# Import necessary libraries
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from skimage.filters import gabor
from sklearn.mixture import GaussianMixture

# Set parameters for feature extraction
SIZE = 224
radius = 3
n_points = 8 * radius
orb = cv2.ORB_create()


def extract_lbp_features(images):
    radius = 3
    n_points = 8 * radius
    lbp_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE)
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype('float')
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        lbp_features.append(lbp_hist)
    return np.array(lbp_features)


def extract_hog_features(images):
    hog_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE)
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
    return np.array(hog_features)



if int(cv2.__version__.split('.')[0]) >= 4:
    sift = cv2.SIFT_create()
else:
    sift = cv2.xfeatures2d.SIFT_create()
    
def extract_sift_features(images):
    sift_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE).astype('uint8')
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors = descriptors.flatten()
            descriptors = descriptors[:SIZE * SIZE]  # Limiter la longueur pour uniformiser
            if len(descriptors) < SIZE * SIZE:
                descriptors = np.pad(descriptors, (0, SIZE * SIZE - len(descriptors)), 'constant')
        else:
            descriptors = np.zeros(SIZE * SIZE)
        sift_features.append(descriptors)
    return np.array(sift_features)



def extract_ldp_features(images):
    ldp_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE)
        ldp = np.zeros_like(img)        
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                center = img[i, j]
                code = 0
                if img[i-1, j-1] >= center: code |= 1 << 0
                if img[i-1, j] >= center: code |= 1 << 1
                if img[i-1, j+1] >= center: code |= 1 << 2
                if img[i, j+1] >= center: code |= 1 << 3
                if img[i+1, j+1] >= center: code |= 1 << 4
                if img[i+1, j] >= center: code |= 1 << 5
                if img[i+1, j-1] >= center: code |= 1 << 6
                if img[i, j-1] >= center: code |= 1 << 7
                ldp[i, j] = code

        ldp_hist, _ = np.histogram(ldp, bins=np.arange(0, 256), range=(0, 255))
        ldp_hist = ldp_hist.astype('float')
        ldp_hist /= (ldp_hist.sum() + 1e-6)
        ldp_features.append(ldp_hist)
    
    return np.array(ldp_features)

    

# Function to extract ORB features
def extract_orb_features(images):
    orb_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE).astype('uint8')
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors = descriptors.flatten()
            descriptors = descriptors[:SIZE * SIZE]  # Limiter la longueur pour uniformiser
            if len(descriptors) < SIZE * SIZE:
                descriptors = np.pad(descriptors, (0, SIZE * SIZE - len(descriptors)), 'constant')
        else:
            descriptors = np.zeros(SIZE * SIZE)
        orb_features.append(descriptors)
    return np.array(orb_features)


# Function to extract BRIEF features
# Use ORB as a replacement for BRIEF keypoint detector
def extract_brief_features(images):
    brief_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE).astype('uint8')
        keypoints = orb.detect(img, None)
        keypoints, descriptors = orb.compute(img, keypoints)
        if descriptors is not None:
            descriptors = descriptors.flatten()
            descriptors = descriptors[:SIZE * SIZE]
            if len(descriptors) < SIZE * SIZE:
                descriptors = np.pad(descriptors, (0, SIZE * SIZE - len(descriptors)), 'constant')
        else:
            descriptors = np.zeros(SIZE * SIZE)
        brief_features.append(descriptors)
    return np.array(brief_features)

# Function to extract Gabor features
def extract_gabor_features(images):
    gabor_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE)
        filt_real, filt_imag = gabor(img, frequency=0.6)
        gabor_features.append(filt_real.flatten())
    return np.array(gabor_features)

def extract_glcm_features(images, distances=[1], angles=[0]):
    glcm_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE).astype('uint8')
        glcm = graycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        asm = graycoprops(glcm, 'ASM').flatten()
        features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm])
        glcm_features.append(features)
    return np.array(glcm_features)


def extract_sift_descriptors(images):
    sift = cv2.SIFT_create()
    all_descriptors = []
    for img in images:
        img = img.reshape(SIZE, SIZE).astype('uint8')
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    return all_descriptors

def form_codebook(descriptors_list, k):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_descriptors)
    return kmeans.cluster_centers_

def compute_vlad(descriptors, codebook):
    k = codebook.shape[0]
    n = codebook.shape[1]
    vlad = np.zeros((k, n), dtype=np.float32)
    
    if descriptors is None or len(descriptors) == 0:
        return vlad.flatten()
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(codebook)
    labels = kmeans.predict(descriptors)
    
    for i in range(k):
        if np.sum(labels == i) > 0:
            vlad[i] = np.sum(descriptors[labels == i] - codebook[i], axis=0)
    
    vlad = vlad.flatten()
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))  # RootSIFT-like normalization
    vlad = vlad / np.linalg.norm(vlad)  # L2 normalization
    return vlad

def extract_vlad_features(images, codebook):
    sift = cv2.SIFT_create()
    vlad_features = []
    for img in images:
        img = img.reshape(SIZE, SIZE).astype('uint8')
        keypoints, descriptors = sift.detectAndCompute(img, None)
        vlad = compute_vlad(descriptors, codebook)
        vlad_features.append(vlad)
    return np.array(vlad_features)



# Load preprocessed data
X_train = np.load('c:/Users/HP/Desktop/sign_verif/datasets/dataset1/X_train.npy')
y_train = np.load('c:/Users/HP/Desktop/sign_verif/datasets/dataset1/y_train.npy')
X_test = np.load('c:/Users/HP/Desktop/sign_verif/datasets/dataset1/X_test.npy')
y_test = np.load('c:/Users/HP/Desktop/sign_verif/datasets/dataset1/y_test.npy')



# Extract features from the training and test data
lbp_train = extract_lbp_features(X_train)
lbp_test = extract_lbp_features(X_test)

hog_train = extract_hog_features(X_train)
hog_test = extract_hog_features(X_test)

sift_train = extract_sift_features(X_train)
sift_test = extract_sift_features(X_test)

ldp_train = extract_ldp_features(X_train)
ldp_test = extract_ldp_features(X_test)

orb_train = extract_orb_features(X_train)
orb_test = extract_orb_features(X_test)

brief_train = extract_brief_features(X_train)
brief_test = extract_brief_features(X_test)

gabor_train = extract_gabor_features(X_train)
gabor_test = extract_gabor_features(X_test)

glcm_train = extract_glcm_features(X_train)
glcm_test = extract_glcm_features(X_test)

sift_descriptors = extract_sift_descriptors(X_train)
codebook = form_codebook(sift_descriptors, k=64)
vlad_train = extract_vlad_features(X_train, codebook)
vlad_test = extract_vlad_features(X_test, codebook)


# Combine all features
X_train_features = np.hstack((lbp_train, hog_train, sift_train, ldp_train, orb_train, gabor_train , glcm_train , vlad_train))
X_test_features = np.hstack((lbp_test, hog_test, sift_test, ldp_test, orb_test, gabor_test,glcm_test ,vlad_test))

# Save extracted features
np.save('c:/Users/HP/Desktop/sign_verif/datasets/dataset1/X_train_features.npy', X_train_features)
np.save('c:/Users/HP/Desktop/sign_verif/datasets/dataset1/X_test_features.npy', X_test_features)



# Combine  ORB, SIFT, and BRIEF features
X_train_Local_Descriptors = np.hstack((sift_train, orb_train, brief_train))
X_test_Local_Descriptors = np.hstack((sift_test, orb_test, brief_test))

np.save('../datasets/dataset5/X_train_Local_Descriptors.npy', X_train_Local_Descriptors)
np.save('../datasets/dataset5/X_test_Local_Descriptors.npy', X_test_Local_Descriptors)


X_train_Texture_Analysis = np.hstack((lbp_train, ldp_train, gabor_train , glcm_train))
X_test_Texture_Analysis = np.hstack((lbp_test, ldp_test, gabor_test,glcm_test ))

np.save('../datasets/dataset5/X_train_Texture_Analysis.npy', X_train_Texture_Analysis)
np.save('../datasets/dataset5/X_test_Texture_Analysis.npy', X_test_Texture_Analysis)


np.save('../datasets/dataset5/X_train_Aggregation_Descriptors.npy', vlad_train)
np.save('../datasets/dataset5/X_test_Aggregation_Descriptors.npy', vlad_test)


np.save('../datasets/dataset5/X_train_Shape_Contour_Analysis.npy', hog_train)
np.save('../datasets/dataset5/X_test_Shape_Contour_Analysis.npy', hog_test)


