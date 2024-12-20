# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:35:25 2024

@author: mursh
"""
import os
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

class RegressionModels:
    def __init__(self, config):
        """
        Initialize the RegressionModels class.
        """
        self.models = {}
        self.config_local = config
        
    def save_models(self):
        """
        Save trained models to the specified directory.

        Parameters:
        - save_dir (str): Path to the directory where models will be saved.
        """
        save_dir = self.config_local['paths']['save_path']
        os.makedirs(save_dir, exist_ok=True)
        for model_name, model in self.models.items():
            file_path = os.path.join(save_dir, f"{model_name}.joblib")
            joblib.dump(model, file_path)
            print(f"Model '{model_name}' saved to {file_path}.")
            
    def load_models(self, name):
        """
        Load trained models from the specified directory.

        Parameters:
        - load_dir (str): Path to the directory containing saved models.
        """
        load_dir = self.config_local['paths']['save_path']
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Directory '{load_dir}' does not exist.")
        
        for file_name in os.listdir(load_dir):
            if file_name.endswith(".joblib"):
                if file_name.split(".")[0]==name:
                    model_name = file_name.split(".")[0]
                    file_path = os.path.join(load_dir, file_name)
                    #self.models[model_name] = joblib.load(file_path)
                    loaded_model = joblib.load(file_path)
                    print(f"Model '{model_name}' loaded from {file_path}.")
                    return loaded_model

    def train_svr(self, X, y, kernel_type):
        """
        Train an SVR model.

        Parameters:
        - X (np.ndarray or pd.DataFrame): Input features.
        - y (np.ndarray or pd.Series): Target variable.
        - kernel (str): Kernel type for SVR ('linear', 'poly', 'rbf', etc.).
        - C (float): Regularization parameter.
        - epsilon (float): Epsilon-tube within which no penalty is associated.
        """
        svr = SVR(kernel=kernel_type, C=self.config_local['model_pars']['C_reg'], epsilon=self.config_local['model_pars']['epsilon'])
        svr.fit(X, y)
        self.models['SVR_{}'.format(kernel_type)] = svr
        print("SVR model trained successfully.")
        self.save_models()

    def train_pls(self, X, y, n_components):
        """
        Train a PLS Regression model.

        Parameters:
        - X (np.ndarray or pd.DataFrame): Input features.
        - y (np.ndarray or pd.Series): Target variable.
        - n_components (int): Number of components to keep.
        """
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y)
        self.models['PLS'] = pls
        print("PLS Regression model trained successfully.")
        self.save_models()

    def train_pcr(self, X, y, n_components):
        """
        Train a PCR model (Principal Component Regression).

        Parameters:
        - X (np.ndarray or pd.DataFrame): Input features.
        - y (np.ndarray or pd.Series): Target variable.
        - n_components (int): Number of principal components to keep.
        """
        # Create a pipeline for PCA followed by Linear Regression
        pca = PCA(n_components=n_components)
        lr = LinearRegression()
        pipeline = Pipeline([('PCA', pca), ('LinearRegression', lr)])
        pipeline.fit(X, y)
        self.models['PCR'] = pipeline
        print("PCR model trained successfully.")
        self.save_models()
    

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a model on test data.

        Parameters:
        - model_name (str): Name of the model to evaluate ('SVR', 'PLS', 'PCR').
        - X_test (np.ndarray or pd.DataFrame): Test features.
        - y_test (np.ndarray or pd.Series): True target values for the test set.

        Returns:
        - metrics (dict): Dictionary containing RMSE and R² scores.
        """            
        model = self.load_models(model_name)    
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metrics = {
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
        }

        print(f"Evaluation metrics for {model_name}: RMSE = {rmse:.4f}, MSE = {mse:.4f}, R² = {r2:.4f}")
        return metrics

    def predict(self, model_name, X):
        """
        Make predictions using a trained model.

        Parameters:
        - model_name (str): Name of the model to use for predictions ('SVR', 'PLS', 'PCR').
        - X (np.ndarray or pd.DataFrame): Input features.

        Returns:
        - predictions (np.ndarray): Predicted values.
        """
        model = self.load_models(model_name)
        predictions = model.predict(X)
        return predictions







