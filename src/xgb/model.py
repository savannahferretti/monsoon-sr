#!/usr/bin/env python

import numpy as np
import xgboost as xgb

class XGBModel:

    def __init__(self, objective='reg:squarederror', n_estimators=100, max_depth=6,
                 learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
                 tree_method='hist', device='cpu', random_state=42, **kwargs):
        '''
        Purpose: Wrapper for XGBoost regressor for precipitation prediction.
        Args:
        - objective (str): loss function to minimize (defaults to 'reg:squarederror')
        - n_estimators (int): number of boosting rounds/trees (defaults to 100)
        - max_depth (int): maximum tree depth (defaults to 6)
        - learning_rate (float): step size shrinkage (defaults to 0.3)
        - subsample (float): fraction of samples to use per tree (defaults to 1.0)
        - colsample_bytree (float): fraction of features to use per tree (defaults to 1.0)
        - tree_method (str): tree construction algorithm (defaults to 'hist')
        - device (str): 'cpu' or 'cuda' for GPU acceleration (defaults to 'cpu')
        - random_state (int): random seed for reproducibility (defaults to 42)
        - **kwargs: additional XGBoost parameters
        '''
        self.params = {
            'objective': objective,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'tree_method': tree_method,
            'device': device,
            'random_state': random_state,
            **kwargs
        }
        self.model = None
        self.evals_result = {}

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds=None, verbose=True):
        '''
        Purpose: Train the XGBoost model with optional early stopping.
        Args:
        - X_train (np.ndarray): training features of shape (n_samples, n_features)
        - y_train (np.ndarray): training targets of shape (n_samples,) or (n_samples, 1)
        - X_valid (np.ndarray): validation features (optional)
        - y_valid (np.ndarray): validation targets (optional)
        - early_stopping_rounds (int): rounds without improvement before stopping (optional)
        - verbose (bool): whether to print training progress (defaults to True)
        Returns:
        - None: trains model in-place
        '''
        # Flatten targets if needed
        if y_train.ndim > 1:
            y_train = y_train.ravel()
        if y_valid is not None and y_valid.ndim > 1:
            y_valid = y_valid.ravel()

        eval_set = [(X_train, y_train)]
        if X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        self.evals_result = self.model.evals_result()

    def predict(self, X):
        '''
        Purpose: Generate predictions using the trained model.
        Args:
        - X (np.ndarray): input features of shape (n_samples, n_features)
        Returns:
        - np.ndarray: predictions of shape (n_samples,)
        '''
        if self.model is None:
            raise RuntimeError('Model has not been trained yet. Call fit() first.')
        return self.model.predict(X)

    def save(self, filepath):
        '''
        Purpose: Save the trained model to a JSON file.
        Args:
        - filepath (str): path to save the model
        Returns:
        - None
        '''
        if self.model is None:
            raise RuntimeError('Model has not been trained yet. Call fit() first.')
        self.model.save_model(filepath)

    def load(self, filepath):
        '''
        Purpose: Load a trained model from a JSON file.
        Args:
        - filepath (str): path to the saved model
        Returns:
        - None: loads model in-place
        '''
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(filepath)
