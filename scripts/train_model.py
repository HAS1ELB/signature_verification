# Import necessary libraries
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import joblib
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import expon
from sklearn.decomposition import PCA
from scipy.stats import randint, uniform
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Load extracted features
X_train = np.load('../datasets/dataset1/X_train_features.npy')
y_train = np.load('../datasets/dataset1/y_train.npy')
X_test = np.load('../datasets/dataset1/X_test_features.npy')
y_test = np.load('../datasets/dataset1/y_test.npy')


# Définir la pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

# Définir la distribution des hyperparamètres
param_dist = {
    'svm__C': expon(scale=100),  # Distribution exponentielle pour C
    'svm__gamma': expon(scale=.1),  # Distribution exponentielle pour gamma
    'svm__kernel': ['rbf', 'linear']  # Choix entre deux kernels
}

# Configuration de RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, refit=True, verbose=2, cv=5, n_jobs=-1)

# Entraîner le modèle avec RandomizedSearchCV
random_search.fit(X_train, y_train) #y_train.ravel()

# Afficher les meilleurs paramètres
print(f'Best Parameters: {random_search.best_params_}')

# Sauvegarder le meilleur modèle
best_model_path = '../models/best_svm_model.pkl'
joblib.dump(random_search.best_estimator_, best_model_path)

print(f'Best model saved to {best_model_path}')



# Define the pipeline with PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=20)),  # Setting a default number of components to reduce the initial computation
    ('knn', KNeighborsClassifier())
])

# Define the hyperparameter distribution with reduced ranges
param_dist = {
    'pca__n_components': randint(15, 30),  # Narrowed range for PCA components
    'knn__n_neighbors': randint(3, 20),  # Narrowed range for number of neighbors
    'knn__weights': ['uniform', 'distance'],  # Choice of weights
    'knn__p': [1, 2],  # Choice between Manhattan and Euclidean distance
    'knn__leaf_size': randint(20, 40),  # Reduced range for leaf size
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Algorithm to use
}

# Configuration of RandomizedSearchCV with reduced iterations and CV folds
n_iter_search = 50  # Reduced number of iterations
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter_search, refit=True, verbose=2, cv=5, n_jobs=-1)

# Train the model with RandomizedSearchCV
random_search.fit(X_train, y_train)

# Print the best parameters
print(f'Best Parameters: {random_search.best_params_}')

# Save the best model
best_model_path = '../models/best_knn_model.pkl'
joblib.dump(random_search.best_estimator_, best_model_path)

print(f'Best model saved to {best_model_path}')




# Define the pipeline with PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),  # Use PCA for dimensionality reduction
    ('rf', RandomForestClassifier())
])

# Define the distribution of hyperparameters
param_dist = {
    'pca__n_components': randint(20, 50),  # Adjust range for PCA components
    'rf__n_estimators': [100, 200, 300, 400, 500],  # Add more options for number of trees
    'rf__max_depth': [None, 10, 20, 30, 40, 50],  # Add more depth options
    'rf__min_samples_split': randint(2, 11),  # Expand range for min_samples_split
    'rf__min_samples_leaf': randint(1, 5),  # Expand range for min_samples_leaf
    'rf__bootstrap': [True, False],  # Use bootstrap samples
    'rf__max_features': ['auto', 'sqrt', 'log2'],  # Add options for max_features
    'rf__criterion': ['gini', 'entropy']  # Criterion for splitting
}

# Configuration of RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=100, refit=True, verbose=2, cv=5, n_jobs=-1, random_state=42)

# Train the model with RandomizedSearchCV
random_search.fit(X_train, y_train.ravel())

# Print the best parameters
print(f'Best Parameters: {random_search.best_params_}')

# Save the best model
best_model_path = '../models/best_rf_model.pkl'
joblib.dump(random_search.best_estimator_, best_model_path)

print(f'Best model saved to {best_model_path}')



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Définir la pipeline avec PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),  # Utiliser PCA pour la réduction de dimension
    ('dt', DecisionTreeClassifier())
])

# Définir la grille des hyperparamètres
param_grid = {
    'dt__max_depth': [None, 10, 20, 30, 40, 50],
    'dt__min_samples_split': [2, 5, 10, 20],
    'dt__min_samples_leaf': [1, 2, 4, 8],
    'dt__criterion': ['gini', 'entropy']
}

# Configuration de GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)

# Entraîner le modèle avec GridSearchCV
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print(f'Best Parameters: {grid_search.best_params_}')

# Sauvegarder le meilleur modèle
best_model_path = '../models/best_dt_model.pkl'
joblib.dump(grid_search.best_estimator_, best_model_path)

print(f'Best model saved to {best_model_path}')





# Définir la pipeline avec PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # Laisser n_components ouvert pour GridSearchCV
    ('logreg', LogisticRegression(max_iter=1000))
])

# Définir la grille des hyperparamètres
param_grid = {
    'pca__n_components': [5, 10, 15, 20, 30],  # Différents nombres de composants pour PCA
    'logreg__C': [0.01, 0.1, 1, 10, 100],  # Paramètre de régularisation
    'logreg__penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Type de régularisation
    'logreg__solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],  # Solvers disponibles pour LogisticRegression
    'logreg__l1_ratio': [0, 0.5, 1]  # Spécifique à elasticnet, ignorez pour les autres
}

# Configuration de GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)

# Entraîner le modèle avec GridSearchCV
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print(f'Best Parameters: {grid_search.best_params_}')

# Sauvegarder le meilleur modèle
best_model_path = '../models/best_logreg_model.pkl'
joblib.dump(grid_search.best_estimator_, best_model_path)

print(f'Best model saved to {best_model_path}')





# Réduire la taille de l'ensemble d'entraînement pour la recherche d'hyperparamètres
X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=42)

# Fonction d'objectif pour Optuna
def objective(trial):
    classifier_name = trial.suggest_categorical('classifier', ['SVM', 'KNN', 'RF', 'DT', 'LogReg'])
    
    if classifier_name == 'SVM':
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=trial.suggest_int('pca_n_components', 5, 50))),
            ('svm', SVC(probability=True, C=trial.suggest_loguniform('svm_C', 1e-3, 1e1), gamma=trial.suggest_loguniform('svm_gamma', 1e-4, 1e-1)))
        ])
        score = cross_val_score(svm_pipeline, X_train_sub, y_train_sub.ravel(), n_jobs=-1, cv=3).mean()
        
    elif classifier_name == 'KNN':
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=trial.suggest_int('knn_n_neighbors', 3, 10)))
        ])
        score = cross_val_score(knn_pipeline, X_train_sub, y_train_sub.ravel(), n_jobs=-1, cv=3).mean()
        
    elif classifier_name == 'RF':
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=trial.suggest_int('rf_n_estimators', 100, 200),
                                          max_depth=trial.suggest_int('rf_max_depth', 10, 50)))
        ])
        score = cross_val_score(rf_pipeline, X_train_sub, y_train_sub.ravel(), n_jobs=-1, cv=3).mean()
        
    elif classifier_name == 'DT':
        dt_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('dt', DecisionTreeClassifier(max_depth=trial.suggest_int('dt_max_depth', 10, 50),
                                          min_samples_split=trial.suggest_int('dt_min_samples_split', 2, 10)))
        ])
        score = cross_val_score(dt_pipeline, X_train_sub, y_train_sub.ravel(), n_jobs=-1, cv=3).mean()
        
    else:
        logreg_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(C=trial.suggest_loguniform('logreg_C', 1e-3, 1e1), 
                                          solver=trial.suggest_categorical('logreg_solver', ['lbfgs', 'liblinear'])))
        ])
        score = cross_val_score(logreg_pipeline, X_train_sub, y_train_sub.ravel(), n_jobs=-1, cv=3).mean()
    
    return score

# Optimisation avec Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Extraire les meilleurs paramètres
best_trial = study.best_trial
best_params = best_trial.params
classifier_name = best_params.pop('classifier')

# Créer le meilleur modèle basé sur les résultats de l'optimisation
if classifier_name == 'SVM':
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=best_params['pca_n_components'])),
        ('svm', SVC(probability=True, C=best_params['svm_C'], gamma=best_params['svm_gamma']))
    ])
    
elif classifier_name == 'KNN':
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=best_params['knn_n_neighbors']))
    ])
    
elif classifier_name == 'RF':
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=best_params['rf_n_estimators'], max_depth=best_params['rf_max_depth']))
    ])
    
elif classifier_name == 'DT':
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeClassifier(max_depth=best_params['dt_max_depth'], min_samples_split=best_params['dt_min_samples_split']))
    ])
    
else:
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(C=best_params['logreg_C'], solver=best_params['logreg_solver']))
    ])

# Entraîner le VotingClassifier avec les meilleurs modèles
voting_clf = VotingClassifier(
    estimators=[('best', best_pipeline)],
    voting='soft'
)

# Entraîner le VotingClassifier sur l'ensemble complet de données
voting_clf.fit(X_train, y_train.ravel())

# Prédire et évaluer
y_pred = voting_clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


