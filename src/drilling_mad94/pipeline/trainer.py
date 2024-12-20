# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:58:13 2024

@author: mursh
"""
import sys
sys.path.append('D:/ML/Drilling_project/src')
from drilling_mad94.models.regression_models import RegressionModels
from drilling_mad94.pipeline.data_processor import Data_Processor


class ModelTrainer:
    def __init__(self, config):
        self.config_local = config
        self.processor = Data_Processor(self.config_local)
        self.reg_model = RegressionModels(self.config_local)
        
    def model_trainer(self, model_type, kernel_type, n_components):
        self.processor.data_read(self.config_local['parameters']['data_type'])
        data = self.processor.data
        target = data[self.config_local['train_pars']['target_variable']]
        features = data.drop(self.config_local['train_pars']['target_variable'], axis=1)
        data = self.processor.scale_data(features, target, self.config_local['train_pars']['test_size'])
        
        if model_type == 'svr':
            self.reg_model.train_svr(data[0], data[2], kernel_type)
            
        if model_type == 'pls':
            self.reg_model.train_pls(data[0], data[2], n_components)
            
        if model_type =='pcr':
            self.reg_model.train_pcr(data[0], data[2], n_components)
        
    




