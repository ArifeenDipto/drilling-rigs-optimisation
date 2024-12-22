# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:59:57 2024

@author: mursh
"""

from drilling_mad94.models.regression_models import RegressionModels
from drilling_mad94.pipeline.data_processor import Data_Processor
from drilling_mad94.utils.display import Plotter


class ModelEvaluator:
    def __init__(self, config):
        self.config_local = config
        self.processor = Data_Processor(self.config_local)
        self.reg_model = RegressionModels(self.config_local)
        self.plot = Plotter(self.config_local)
        self.data = None
        self.features =None
        self.target =None
        self.target_vars = None
        self.prediction = None
        
    def create_data(self):
        self.processor.data_read(self.config_local['parameters']['data_type'])
        data = self.processor.data
        target = data[self.config_local['train_pars']['target_variable']]
        self.target = target
        features = data.drop(self.config_local['train_pars']['target_variable'], axis=1)
        self.features = features
        data = self.processor.scale_data(features, target, self.config_local['train_pars']['test_size'])
        self.data = data
        
    def model_evaluator(self, model_type): 
        self.create_data()
        self.reg_model.evaluate_model(model_type, self.data[1], self.data[3])
        
    def model_prediction(self, model_type):
        self.create_data()
        predictors = self.processor.scale_whole_data(self.features)
        self.target_vars = self.processor.scale_whole_data(self.target.values.reshape(-1,1))
        self.prediction = self.reg_model.predict(model_type, predictors)
    
        
    def plot_prediction(self, model_name):
        self.model_prediction(model_name)     
        self.plot.plot_predictions(self.target_vars, self.prediction)
        
    def model_analysis(self, model_name):
        self.create_data()
        #pred = self.reg_model.predict(model_name, self.data[1])
        if model_name == 'SVR_linear' or 'SVR_rbf' or 'SVR_poly' or 'SVR_sigmoid':
            support_vectors = self.reg_model.svr_properties(model_name, self.data[0], self.data[2])
            self.plot.plot_support_vector(self.data[0], self.config_local['parameters']['feature_1'], self.config_local['parameters']['feature_2'], support_vectors[0])
            
        
    def plot_residual(self, model_name):
        self.create_data()
        predictors = self.processor.scale_whole_data(self.features)
        target_vars = self.processor.scale_whole_data(self.target.values.reshape(-1,1))
        prediction = self.reg_model.predict(model_name, predictors)
        residuals = target_vars - prediction
        self.plot.plot_residuals(target_vars, prediction)       
        