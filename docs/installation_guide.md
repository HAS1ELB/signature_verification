# Installation Guide

This guide will help you set up the environment and install the necessary dependencies to run the signature verification project.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/signature_verification.git
    cd signature_verification
    ```

2. **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset**:

    Place your training and testing data in the `sign_data/train` and `sign_data/test` directories, respectively. Ensure `train_data.csv` and `test_data.csv` are in the `sign_data` directory.

5. **Run the notebooks**:

    Launch Jupyter Notebook and open the notebooks in the `notebooks` directory to see step-by-step instructions.

    ```bash
    jupyter notebook
    ```

## Additional Notes

- Make sure to keep your virtual environment activated when working on the project.
- If you encounter any issues, refer to the [README](README.md) or open an issue in the repository.
