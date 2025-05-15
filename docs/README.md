# Signature Verification Project

This project aims to verify signatures using machine learning techniques. It includes scripts for preprocessing images, extracting features, and training a model.

## Project Structure

signature_verification/
├── datasets/
│ ├── dataset1/
│     ├── train/
│     ├── test/
│     ├── X_train.npy
│     ├── X_test.npy
│     ├── y_test.npy
│     ├── y_train.npy
│     ├── X_train_features.npy
│     ├── X_test_features.npy
│ ├── dataset2/
│     ├── train/
│     ├── test/
│     ├── X_train.npy
│     ├── X_test.npy
│     ├── y_test.npy
│     ├── y_train.npy
│     ├── X_train_features.npy
│     ├── X_test_features.npy
│ ├── dataset3/
│     ├── train/
│     ├── test/
│     ├── X_train.npy
│     ├── X_test.npy
│     ├── y_test.npy
│     ├── y_train.npy
│     ├── X_train_features.npy
│     ├── X_test_features.npy
│ ├── dataset4/
│     ├── train/
│     ├── test/
│     ├── X_train.npy
│     ├── X_test.npy
│     ├── y_test.npy
│     ├── y_train.npy
│     ├── X_train_features.npy
│     ├── X_test_features.npy
│ ├── dataset5/
│     ├── train/
│     ├── test/
│     ├── X_train.npy
│     ├── X_test.npy
│     ├── y_test.npy
│     ├── y_train.npy
│     ├── X_train_features.npy
│     ├── X_test_features.npy
├── models/
│ └── best_svm_model.pkl
│ └── best_knn_model.pkl
│ └── best_logreg_model.pkl
│ └── best_dt_model.pkl
│ └── best_rf_model.pkl
│ └── best_voting_model.pkl
├── scripts/
│ ├── preprocess_images.py
│ ├── extract_features.py
│ ├── train_model.py
│ ├── voting_Classifier.py
├── notebooks/
│ ├── data_preparation.ipynb
│ ├── feature_extraction.ipynb
│ ├── model_training.ipynb
│ ├── evaluation.ipynb
│ ├── voting_Classifier.ipynb
├── myenv/
├── docs/
│ ├── README.md
│ ├── installation_guide.md
│ └── architecture.md
└── tests/
│  ├── test_preprocess_images.py
│  ├── test_extract_features.py
│  └── test_train_model.py


## Getting Started

1. Clone the repository.
2. Follow the instructions in the [installation guide](docs/installation_guide.md).
3. Run the preprocessing, feature extraction, and model training scripts as explained in the notebooks.

## Usage

- **Preprocess Images**: `python scripts/preprocess_images.py`
- **Extract Features**: `python scripts/extract_features.py`
- **Train Model**: `python scripts/train_model.py`

## Contributing

Please read the [contributing guidelines](docs/contributing.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License.
