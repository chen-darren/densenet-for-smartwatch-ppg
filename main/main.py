# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:52:53 2024

@author: dchen
"""
import torch
import torch.nn as nn
import time
import datetime as dt
import os
import sys

# Import my own functions and classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pathmaster import PathMaster
import utils.model_func as model_func
import utils.dataloader as dataloader
import utils.dataloader_database as dataloader_database
import utils.dataloader_combined as dataloader_combined

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Device and drives
    is_linux = False
    is_hpc = False
    is_internal = False
    is_external = False
    binary = False
    
    # Input
    is_tfs = True
    
    # Dataset combination specifics
    combination = 'combined_dataset'
    split = 'multiclass'
    
    # Database
    database = 'mimic3'
    
    # Intialize the focus
    # focus = 'misc'
    # focus = '0_image_resolution'
    # focus = '6_dropout'
    focus = 'thesis_results_final_pulsewatch_multiclass'
    
    # Initialize the file tag
    # file_tag = 'misc'
    # file_tag = 'batch64_worker6_prefetch2_epoch20'
    file_tag = 'Proposed_Model'
    # file_tag = 'Pulsewatch'
    
    # # For str(dt.dateitme.now())
    # # Define characters to replace with underscores
    # chars_to_replace = [' ', ':', '.', '-']
    
    # # Replace characters with underscores
    # for char in chars_to_replace:
    #     file_tag = file_tag.replace(char, '_')
    
    # Image resolution
    # img_res = '256x256'
    img_res = '128x128_float16'
    
    # Data type: the type to convert the data into when it is loaded in
    data_type = torch.float32
    
    # Model type
    model_type = torch.float32 # Sets the data type of the entire model...
    
    # Create a PathMaster object
    pathmaster = PathMaster(is_linux, is_hpc, is_tfs, is_internal, is_external, focus, file_tag, img_res)

    # Image dimensions
    img_channels = 1
    img_size = 128 
    downsample = None
    standardize = True
    
    # Run parameters
    n_epochs = 100
    if binary:
        n_classes = 2
    else:
        n_classes = 3
    patience = round(n_epochs / 10) if n_epochs > 50 else 5
    # patience = 100
    save = True
    
    # Split UIDs
    # train_set, val_set, test_set = dataloader.split_uids(pathmaster)
    # train_set, val_set, test_set = dataloader.split_uids_60_10_30(pathmaster)
    train_set, val_set, test_set = dataloader.split_uids_60_10_30_v2(pathmaster)
    # train_set, val_set, test_set = dataloader.split_uids_60_10_30_balanced(pathmaster)
    # train_set, val_set, test_set = dataloader.split_uids_60_10_30_noPACPVC(pathmaster)
    
    # Data loading details
    data_format = 'pt'
    batch_size = 256
    
    # # Preprocess data (general)
    # train_loader, val_loader, _ = dataloader.preprocess_data(data_format, train_set, val_set, test_set, 
    #                                               batch_size, standardize, False, img_channels, img_size, downsample, data_type, pathmaster, binary)
    train_loader, val_loader, test_loader = dataloader.preprocess_data(data_format, train_set, val_set, test_set, 
                                                  batch_size, standardize, False, img_channels, img_size, downsample, data_type, pathmaster, binary)
    # _, _, test_loader = dataloader.preprocess_data(data_format, train_set, val_set, test_set, 
    #                                               batch_size, standardize, False, img_channels, img_size, downsample, data_type, pathmaster, binary)
    
    # # Preprocess database data
    # test_loader = dataloader_database.preprocess_data(database, batch_size, standardize, img_channels, img_size, 
    #                                                 downsample, data_type, pathmaster, binary)
    
    # # Preprocess split dataset
    # train_loader, val_loader, test_loader = dataloader_database.preprocess_data_split(database, batch_size, standardize, img_channels, img_size, 
    #                                                               downsample, data_type, pathmaster, binary, test=True)
    
    # # Preprocess combined dataset
    # train_loader, val_loader = dataloader_combined.preprocess_data(combination, split, batch_size, standardize, img_channels, img_size,
    #                                                                downsample, data_type, pathmaster, binary)
    
    # # Training and validation ============================================================================================================================
    # # Create model hyperparameters
    config = { # Proposed model
        'num_layers_per_dense': 4,
        'growth_rate': 16,
        'compression': 0.8,
        'bottleneck': False,
        'drop_rate': 0.1,
        'class_weights': [59743/38082, 59743/13800, 59743/7861], # [59743/(38082+7861), 59743/13800],
        'learning_rate': 0.0000215443469003188,
        'lambda_l1': 0.0000774263682681127,
        'activation': nn.GELU(),
        }
    
    start_time = time.time()
    model_func.train_validate_DenseNet_config(config, train_loader, val_loader, n_epochs, 
                                              n_classes, patience, save, pathmaster)

    end_time = time.time()
    time_passed = end_time-start_time # in seconds
    time_passed = time_passed / 60 # in minutes
    # time_passed = time_passed / 3600 # in hours
    print('\nTraining and validation took %.2f' % time_passed, 'minutes')
    
    # # Run best model ===================================================================================================================================
    start_time = time.time()
    if not binary:
        model_func.best_DenseNet_config(test_loader, model_type, n_classes, save, pathmaster)
    else:
        model_func.best_DenseNet_config_binary(test_loader, model_type, n_classes, save, pathmaster)
    end_time = time.time()
    time_passed = end_time-start_time # in seconds
    # time_passed = time_passed / 60 # in minutes
    # time_passed = time_passed / 3600 # in hours
    print('\nTesting took %.2f' % time_passed, 'seconds')
    # ===================================================================================================================================================
 
if __name__ == '__main__':
    main()