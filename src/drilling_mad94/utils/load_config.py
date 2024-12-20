# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:05:01 2024

@author: mursh
"""
import os
import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
