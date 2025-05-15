import pytest
import os
import numpy as np
from scripts.preprocess_images import preprocess_image, load_images_from_directory, show_random_images

# Set image size
SIZE = 224

def test_preprocess_image_valid_path():
    # Test preprocessing an image with a valid path
    valid_img_path = 'c:/Users/HP/Desktop/sign_verif/datasets/dataset1/train/001/001_01.png' 
    processed_img = preprocess_image(valid_img_path)
    
    # Verify the image is processed correctly
    assert processed_img is not None, "Processed image should not be None"
    assert processed_img.shape == (SIZE, SIZE), f"Processed image size should be {SIZE}x{SIZE}"
    assert processed_img.dtype == np.uint8, "Processed image dtype should be uint8"

def test_preprocess_image_invalid_path():
    # Test preprocessing an image with an invalid path
    invalid_img_path = 'c:/Users/HP/Desktop/sign_verif/datasets/dataset1/train/invalid_path.png'
    with pytest.raises(Exception):
        preprocess_image(invalid_img_path)

def test_load_images_from_directory():
    # Test loading and preprocessing images from a directory
    train_dir = 'c:/Users/HP/Desktop/sign_verif/datasets/dataset1/train'  
    real_images, forged_images = load_images_from_directory(train_dir)
    
    # Verify the images are loaded and processed correctly
    assert len(real_images) > 0, "There should be real images in the directory"
    assert len(forged_images) > 0, "There should be forged images in the directory"
    assert real_images[0].shape == (SIZE, SIZE), f"Real image size should be {SIZE}x{SIZE}"
    assert forged_images[0].shape == (SIZE, SIZE), f"Forged image size should be {SIZE}x{SIZE}"

def test_show_random_images(monkeypatch):
    # Test displaying random images
    real_images = np.random.randint(0, 255, (10, SIZE, SIZE), dtype=np.uint8)
    real_labels = np.zeros((10, 1))
    
    # Mock plt.show to avoid actual display during tests
    def mock_show():
        pass
    
    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    
    show_random_images(real_images, real_labels, "Random Real Images")

if __name__ == "__main__":
    pytest.main()
