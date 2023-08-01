# Dicts of hyperparameters to use in training as a grid search.


RANDOM_FOREST = {
    'max_depth': [10, 20, 30, 50],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [25, 50, 75],
    'min_samples_leaf': [25, 50, 75],
    'n_estimators': [100, 150, 200]
}

LIGHTGBM = {
    'boosting_type':['gbdt','dart','goss'],
    'learning_rate': [0.1, 0.01, 0.02,0.001, 0.002],
    'n_estimators': [100, 150, 200, 300, 400],
    'max_depth':[-1,50,100,200,250, 400]    
}

ENSSEMBLE = {
    # Lightgbm
    'lg__boosting_type':['gbdt','dart','goss'],
    'lg__learning_rate': [0.1, 0.01,0.02,0.001, 0.002],
    'lg__n_estimators': [100, 150, 200, 300, 400],
    'lg__max_depth':[-1,50,100,200,250, 400],
    #Random Forest
    'rf__max_depth': [10, 20, 30, 50],
    'rf__max_features': ['auto', 'sqrt', 'log2'],
    'rf__min_samples_split': [25, 50, 75],
    'rf__min_samples_leaf': [25, 50, 75],
    'rf__n_estimators': [100, 150, 200],
    #XGBoost
    'xgb__max_depth': [3,6,10],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__n_estimators': [100, 500, 1000],
    'xgb__colsample_bytree': [0.3, 0.7],
    #CATBoost
    'cat__learning_rate':[0.1,0.01,0.02],
    'cat__max_depth':[3,6,10],
    'cat__iterations':[1000,2000,5000],
    'cat__depth':[3,6,9]
      
}

XGBOOST = {
    'max_depth': [3,6,10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'colsample_bytree': [0.3, 0.7]

}

CATBOOST = {
    'learning_rate':[0.1,0.01,0.02],
    'max_depth':[3,6,10],
    'iterations':[1000,2000,5000],
    'depth':[3,6,9]
    }