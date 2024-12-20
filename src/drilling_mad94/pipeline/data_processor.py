# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:11:07 2024

@author: mursh
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from drilling_mad94.utils.display import Plotter
   
class Data_Processor():
    def __init__(self, config):
                
        self.data_path = config['paths']['raw_data']
        self.proc_path = config['paths']['proc_data']
        self.corr_type = config['parameters']['correlation_type']
        self.corr_th = config['parameters']['correlation_threshold']
        self.target_var =  config['parameters']['target_feature']
        self.data = None
        self.config_local = config
        
        
    def read_file(self, path):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            logging.error(f"File not found: {path}")
            sys.exit(1)
            
    def data_read(self, data_type):
        logging.info(f"Reading {data_type} data...")
        if data_type == 'raw':
            self.data = self.read_file(self.data_path)
        elif data_type == 'processed':
            self.data = self.read_file(os.path.join(self.proc_path, 'selected_data.csv'))
            
    def show_correlation(self, data_type):
        logging.info(f"Showing correlation matrix for {data_type} data...")
        self.data_read(data_type)
        plotter = Plotter(self.config_local)
        plotter.plot_correlation_matrix(self.data)
        
    #Correlation based feature selection
    def select_features(self, data_type):
        file_name = 'selected_data.csv'
        logging.info(f"Selecting features from {data_type} data...")
        self.data_read(data_type)
        correlation_matrix = self.data.corr(self.corr_type)
        target_correlations = correlation_matrix[self.target_var]
        selected_features = target_correlations[abs(target_correlations) > self.corr_th].index.tolist()
        selected_features.remove(self.target_var)
        selected_features_with_target = selected_features + [self.target_var]
        selected_data = self.data[selected_features_with_target]
        selected_data.to_csv(os.path.join(self.proc_path, file_name), index=False)
        
    #train test split
    def split_data(self, X, y, split_ratio):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split_ratio, random_state=42)
        return X_tr, X_te, y_tr, y_te
        
    #Feature scaling
    def scale_data(self, X, y, split_ratio):
        logging.info(f"Scaling the data using MinMaxScaler")
        scaler = MinMaxScaler()
        scaler1 = MinMaxScaler()
        exp_data = self.split_data(X, y, split_ratio)
        
        tr_X = scaler.fit(exp_data[0])
        tr_X = scaler.transform(exp_data[0])       
        te_X = scaler.transform(exp_data[1])
        
        tr_y = scaler1.fit(exp_data[2].values.reshape(-1, 1))
        tr_y = scaler1.transform(exp_data[2].values.reshape(-1, 1))
        te_y = scaler1.transform(exp_data[3].values.reshape(-1, 1))
        
        return tr_X, te_X, tr_y.ravel(), te_y.ravel()
    
    def scale_whole_data(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data
        
        
        
            
        
    
    
    

    
    
    