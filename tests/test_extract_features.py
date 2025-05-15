import pytest
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from scripts.extract_features import (
    extract_lbp_features, extract_hog_features, extract_sift_features, extract_ldp_features,
    visualize_lbp_features, visualize_hog_features, visualize_sift_features, visualize_ldp_features,
    visualize_lbp_histogram, visualize_hog_histogram, visualize_sift_histogram, visualize_ldp_histogram
)

# Constants
SIZE = 224
radius = 3
n_points = 8 * radius
orb = cv2.ORB_create()

# Test data
test_image = np.random.randint(0, 255, (SIZE, SIZE), dtype=np.uint8)
test_images = np.array([test_image for _ in range(5)])

def test_extract_lbp_features():
    lbp_features = extract_lbp_features(test_images)
    assert lbp_features.shape == (5, n_points + 2), "La taille des caractéristiques LBP n'est pas correcte"

def test_extract_hog_features():
    hog_features = extract_hog_features(test_images)
    assert hog_features.shape[0] == 5, "Le nombre de caractéristiques HOG doit correspondre au nombre d'images"
    assert hog_features[0].shape[0] > 0, "La taille des caractéristiques HOG doit être supérieure à 0"

def test_extract_sift_features():
    sift_features = extract_sift_features(test_images)
    assert sift_features.shape == (5, SIZE * SIZE), "La taille des caractéristiques SIFT n'est pas correcte"

def test_extract_ldp_features():
    ldp_features = extract_ldp_features(test_images)
    assert ldp_features.shape == (5, 255), "La taille des caractéristiques LDP n'est pas correcte"

def test_visualize_lbp_features(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_lbp_features(test_images, index=0)

def test_visualize_hog_features(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_hog_features(test_images, index=0)

def test_visualize_sift_features(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_sift_features(test_images, index=0)

def test_visualize_ldp_features(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_ldp_features(test_images, index=0)

def test_visualize_lbp_histogram(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_lbp_histogram(test_images, index=0)

def test_visualize_hog_histogram(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_hog_histogram(test_images, index=0)

def test_visualize_sift_histogram(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_sift_histogram(test_images, index=0)

def test_visualize_ldp_histogram(monkeypatch):
    def mock_show():
        pass

    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    visualize_ldp_histogram(test_images, index=0)

if __name__ == "__main__":
    pytest.main()
