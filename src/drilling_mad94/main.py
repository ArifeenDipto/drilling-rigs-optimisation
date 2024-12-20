# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:09:36 2024

@author: mursh
"""
import os
import sys
import yaml
import argparse
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.data_processor import Data_Processor
from utils.load_config import load_config
from utils.display import Plotter
from pipeline.trainer import ModelTrainer
from pipeline.evaluator import ModelEvaluator

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Customize the output format
    datefmt="%Y-%m-%d %H:%M:%S"  # Customize the date format
)

def parse_args(config):
    parser = argparse.ArgumentParser(description="Data processing CLI with YAML configuration.")
    parser.add_argument('--config', type=str, default=config,
                        help="Path to the YAML configuration file (default: %(default)s).")
    parser.add_argument('--task', choices=['read', 'plot', 'select_features', 'train', 'test'], required=True,
                        help="Task to perform: read data or select features.")
    parser.add_argument('--data_type', type=str, 
                        help="Type of data: raw or processed.")
    parser.add_argument('--model', type=str,
                        help='Select the type of models to train')
    parser.add_argument('--kernel', type=str,
                        help='Type of Kernel for SVR model.')
    parser.add_argument('--n_comp', type=int,
                        help='Number of components for PLS and PCR model')
    parser.add_argument('--plot_type', type=str, choices=['correlation', 'predictions', 'residuals'],
                        help="Type of plot to generate: 'correlation', 'predictions', 'residuals'.")
    
    
    args = parser.parse_args()
    
    
    
    if args.task in ['read', 'select_features'] and not args.data_type:
        parser.error("--data_type is required for the selected task.")
        
    if args.task == 'train':
        if not args.model:
            parser.error("Select the model type")
        if not args.kernel:
            parser.error("--kernel is required for the 'train' task.")
        if not args.n_comp:
            parser.error("--number of components is required for training PLS and PCR")
            
    if args.task == 'test':
        if not args.model:
            parser.error("Select the model type")
            
    if args.task == 'plot' and not args.plot_type:
        parser.error("--plot_type is required for the 'plot' task.")
        
    return args


def main():
    config = load_config()
    
    args = parse_args(config) # Parse command-line arguments

    processor = Data_Processor(config=args.config) # Instantiate the Data_Processor class
    trainer = ModelTrainer(config=args.config)
    evaluator = ModelEvaluator(config=args.config)
    # Perform tasks based on the command-line argument
    if args.task == 'read':
        print("\n What type of data you want to read? : raw or processed?")
        choice = input('\n Type 1 for raw and 2 for processed; Answer: ').strip()
        if choice == '1':
            processor.data_read('raw')
            #print(f"Data loaded successfully from {processor.data_path}")
        else:
            processor.data_read('processed')
            #print(f"Data loaded successfully from {processor.proc_path}")
            
    elif args.task == 'plot':
        if args.plot_type == 'correlations':
            processor.show_correlation(args.data_type)
        elif args.plot_type == 'predictions':
            evaluator.plot_prediction(args.model)
                   
    elif args.task == 'select_features':
        processor.select_features(args.data_type)
    elif args.task == 'train':
        trainer.model_trainer(args.model, args.kernel, args.n_comp)
    elif args.task == 'test':
        evaluator.model_evaluator(args.model)
        

if __name__ == "__main__":
    main()
    
