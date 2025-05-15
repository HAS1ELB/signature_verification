import pytest
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scripts.train_model import train_model, save_model  # Assurez-vous que le chemin est correct
import numpy as np

class DummyModel:
    def __init__(self, model_type='SVM'):
        self.model_type = model_type
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=5)),  # Réduire le nombre de composants pour le test
            (model_type.lower(), self._get_model_instance(model_type))
        ])
    def _get_model_instance(self, model_type):
        if model_type == 'SVM':
            return SVC(probability=True)
        elif model_type == 'KNN':
            return KNeighborsClassifier()
        elif model_type == 'RF':
            return RandomForestClassifier()
        elif model_type == 'DT':
            return DecisionTreeClassifier()
        elif model_type == 'LogReg':
            return LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    def fit(self, X, y):
        self.pipeline.fit(X, y)
    def predict(self, X):
        return self.pipeline.predict(X)
    
def test_save_model(tmp_path):
    # Test de sauvegarde du modèle
    model = DummyModel()  # Utiliser un modèle Dummy pour le test
    save_path = tmp_path / "model.pkl"
    save_model(model, save_path)
    
    # Vérifier que le fichier du modèle est créé
    assert os.path.exists(save_path), "Le fichier du modèle doit être créé"


def test_train_model():
    # Test de base
    train_data_path = '../datasets/dataset1/X_train_features.npy'  
    labels_path = '../datasets/dataset1/y_train.npy'  

    # Charger les données
    X_train = np.load(train_data_path)
    y_train = np.load(labels_path)

    model = train_model(X_train, y_train)
    
    # Vérifier que le modèle est entraîné correctement
    assert model is not None, "Le modèle entraîné ne doit pas être None"
    assert hasattr(model, 'predict'), "Le modèle doit avoir une méthode 'predict'"


def test_dummy_model():
    # Test des prédictions avec le DummyModel
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    
    model = DummyModel(model_type='SVM')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Vérifier que les prédictions sont générées correctement
    assert predictions is not None, "Les prédictions ne doivent pas être None"
    assert len(predictions) == len(X_test), "Le nombre de prédictions doit correspondre à la taille de l'ensemble de test"

