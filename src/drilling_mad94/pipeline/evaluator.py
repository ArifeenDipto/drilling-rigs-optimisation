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
        
    def plot_prediction(self, model_name):
        self.create_data()
        predictors = self.processor.scale_whole_data(self.features)
        target_vars = self.processor.scale_whole_data(self.target.values.reshape(-1,1))
        prediction = self.reg_model.predict(model_name, predictors)
        self.plot.plot_predictions(target_vars, prediction)
        
        
        