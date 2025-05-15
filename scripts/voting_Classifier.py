import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Load extracted features
X_train = np.load('../datasets/dataset5/X_train_Shape_Contour_Analysis.npy')
y_train = np.load('../datasets/dataset5/y_train.npy')
X_test = np.load('../datasets/dataset5/X_test_Shape_Contour_Analysis.npy')
y_test = np.load('../datasets/dataset5/y_test.npy')




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


# Sauvegarder le meilleur modèle
best_model_path = '../models/best_voting_model.pkl'
joblib.dump(voting_clf, best_model_path)
print(f'Best model saved to {best_model_path}')