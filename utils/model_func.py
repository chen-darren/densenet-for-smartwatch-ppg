# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:58:20 2024

@author: dchen
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import time
from torch.cuda.amp import autocast, GradScaler

# Import my own functions and classes
from utils import plot_save_func
from models.densenet_configurable import DenseNet as DenseNet_config

# If GPU is available, use GPU, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
       
       
def train_validate_DenseNet_config(config, train_loader, val_loader,
                            n_epochs=100, n_classes=3, patience=10, save=False, 
                            pathmaster=None):
    # # Set filetag
    # file_tag = str(dt.datetime.now())
    # # Define characters to replace with underscores
    # chars_to_replace = [' ', ':', '.', '-']
    
    # # Replace characters with underscores
    # for char in chars_to_replace:
    #     file_tag = file_tag.replace(char, '_')
    # pathmaster.set_file_tag(file_tag)
    
    # Save hyperparameters
    model_hyperparameters = { # Default, no bottleneck or compression
        'num_layers_per_dense': config['num_layers_per_dense'],
        'growth_rate': config['growth_rate'],
        'compression': config['compression'],
        'bottleneck': config['bottleneck'],
        'drop_rate': config['drop_rate'],
        'class_weights': config['class_weights'],
        'learning_rate': config['learning_rate'],
        'lambda_l1': config['lambda_l1'],
        'activation': activation_to_string(config['activation']),
        }
    
    if save:
            plot_save_func.save_hyperparameters(model_hyperparameters, pathmaster)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_channels = 1
    
    model = DenseNet_config(img_channels, config['num_layers_per_dense'], n_classes, config['growth_rate'], config['compression'], 
                     config['bottleneck'], config['drop_rate'], config['activation']).to(device=device)
    
    # Loss function and optimizer
    criterion_train = nn.CrossEntropyLoss(weight=torch.tensor(config['class_weights']).to(device=device))
    criterion_val = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = IdentityScheduler(optimizer)
        
    
    # Scalers
    scaler = GradScaler()
    
    # Initialize losslists
    losslist_train = []
    losslist_val = []
    
    # Initialize predictions lists
    predictions_list_train = []
    predictions_list_val = []
    
    # Initialize true labels lists
    true_labels_list_train = []
    true_labels_list_val = []
    
    # Initialize runtime list
    runtime_list = []
    
    # Create EarlyStoppingCallback object
    early_stopping_callback = EarlyStoppingCallback(patience)
    
    # Initialize best validation loss
    best_loss_val = float('inf') # If no checkpoint is loaded, set to infinity
    
    start_epoch = 0
    # Training and validation
    print('\n===========================================================================================')
    sys.stdout.flush()
    for epoch in range(start_epoch, n_epochs): # Creates a training progress bar with units of epoch
        start_time = time.time()
        sys.stderr.flush()
        print("\nEntering Epoch:", epoch)    
        # Training
        model.train()
        # Reset training sum of epoch loss and batch_count
        sum_epoch_loss_train = 0
        sys.stdout.flush()
        
        # Epoch predictions
        predictions_epoch_train = []
        predictions_epoch_val = []
        
        for train_batch in tqdm(train_loader, total=len(train_loader), desc='Training Epoch', unit='batch', leave=False):
            # Extract input and labels
            # train_batch['data'].shape = [batch_size, img_channels, img_size, img_size]
            X_train = train_batch['data'].reshape(train_batch['data'].shape[0], train_batch['data'].shape[1], train_batch['data'].shape[-1], train_batch['data'].shape[-1]).to(device=device)
            Y_train = train_batch['label'].to(device=device)
            
            if epoch == start_epoch:
                true_labels_list_train.append(torch.reshape(Y_train, (-1,1)))
            
            with autocast():
                # Forward pass
                logits, predictions, _ = model(X_train)
            
            predictions_epoch_train.append(torch.reshape(predictions, (-1,1)))   
            
            # Regularization
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            
            # Calculate sum of total loss for epoch with regularization
            batch_loss_train = criterion_train(logits.to(torch.float32), Y_train.long()) # Criterion returns a scalar tensor
            batch_loss_train += config['lambda_l1'] * l1
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
    
            # Backwards pass
            scaler.scale(batch_loss_train).backward()

            # Optimizer step
            scaler.step(optimizer)
            
            # Scaler update
            scaler.update()
            
            # Generate epoch loss
            sum_epoch_loss_train += batch_loss_train.item()
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch loss for training
        epoch_loss_train = sum_epoch_loss_train / len(train_batch)
        losslist_train.append(epoch_loss_train)
        
        sys.stderr.flush()
        print('\nTraining for Epoch', epoch, 'has been completed!')
        sys.stdout.flush()
        
        # Validation
        model.eval()
        sum_epoch_loss_val = 0
        with torch.no_grad(): # Disable gradient computation during validation
            sys.stdout.flush()
            for val_batch in tqdm(val_loader, total=len(val_loader), desc='Validation Epoch', unit='batch', leave=False):
                # Extract input and labels
                X_val = val_batch['data'].reshape(val_batch['data'].shape[0], val_batch['data'].shape[1], val_batch['data'].shape[-1], val_batch['data'].shape[-1]).to(device=device)
                Y_val = val_batch['label'].to(device=device)
                
                if epoch == start_epoch:
                    true_labels_list_val.append(torch.reshape(Y_val, (-1,1)))
                
                # Forward pass
                logits, predictions, _ = model(X_val)
                predictions_epoch_val.append(torch.reshape(predictions, (-1,1)))

                # Calculate sum of total loss for epoch
                sum_epoch_loss_val += criterion_val(logits.float(), Y_val.long()).item() # Criterion returns a scalar tensor

        # Calculate epoch loss for validation
        epoch_loss_val = sum_epoch_loss_val / len(val_loader)
        losslist_val.append(epoch_loss_val)
        
        sys.stderr.flush()
        print('\nValidation for Epoch', epoch, 'has been completed!')
        sys.stdout.flush()
        
        # Return the best validation loss and save best checkpoint (epoch)
        best_loss_val = save_best_checkpoint(model, optimizer, scheduler, epoch, epoch_loss_val, best_loss_val, pathmaster)
        
        # Update line
        sys.stderr.flush()
        print("\n======> Epoch: {}/{}, Training Loss: {:.4f}, Validation Loss: {:.4f}".format(epoch, n_epochs-1, epoch_loss_train, epoch_loss_val))
        print('\n===========================================================================================')
        sys.stdout.flush()
        
        # Add epoch predictions
        predictions_epoch_train = np.array(torch.cat(predictions_epoch_train, dim=0).to('cpu'))
        predictions_epoch_val = np.array(torch.cat(predictions_epoch_val, dim=0).to('cpu'))
        
        predictions_list_train.append(predictions_epoch_train)
        predictions_list_val.append(predictions_epoch_val)
        
        # Add epoch time to runtime_list
        end_time = time.time()
        time_passed = end_time-start_time # in seconds
        runtime_list.append(time_passed)
        
        # Call the early stopping callback
        if early_stopping_callback(epoch, epoch_loss_val):
            break
    
    # Convert true label list into array
    true_labels_train = np.array(torch.cat(true_labels_list_train, dim=0).to('cpu'))
    true_labels_val = np.array(torch.cat(true_labels_list_val, dim=0).to('cpu'))
    
    if save:
        title = 'Training and Validation Loss'
        plot_save_func.train_val_loss(losslist_train, losslist_val, title, save, pathmaster)
        
        title = 'Training and Validation Accuracy'
        plot_save_func.accuracy_curves(true_labels_train, true_labels_val, predictions_list_train, predictions_list_val, title, save, pathmaster)
        
        plot_save_func.save_losslists(losslist_train, losslist_val, pathmaster)
        plot_save_func.save_runtime_list(runtime_list, pathmaster)
        

def best_DenseNet_config(data_loader, model_type=torch.float32, n_classes=3, save=False, pathmaster=None):
    print('\n===========================================================================================')
    
    # If GPU is available, use GPU, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get paths
    checkpoints_path = pathmaster.checkpoints_path()
    
    # Load model hyperparameters
    num_layers_per_dense, growth_rate, compression, bottleneck, drop_rate, _, _, _, activation = load_hyperparameters_random_search(pathmaster)
    # When testing on the test set, drop_rate, class_weights, learning_rate, and lambda_l1 are not needed
    
    # Initialize model
    model = DenseNet_config(img_channels=1, num_layers_per_dense=num_layers_per_dense, n_classes=n_classes, growth_rate=growth_rate, 
                  compression=compression, bottleneck=bottleneck, drop_rate=drop_rate,
                  activation=activation).to(device=device, dtype=model_type)
    
    # Create criterion for loss
    criterion = nn.CrossEntropyLoss()
    
    # If checkpoint is not specified, terminate the function
    checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_' + pathmaster.file_tag + '.pt')
    assert os.path.exists(checkpoint_path), 'Function terminated. Not a valid checkpoint path.'
    
    # Load model
    model = load_model(model, pathmaster)
    
    # Initialize true label lists
    true_labels_list = []
    
    # Intialize output (prediction) lists
    predictions_list = []
    prediction_proba_list = []
    
    # Initialize segment names list
    segment_names_list = []
    
    # Evaluation
    model.eval()
    cum_loss = 0
    with torch.no_grad(): # Disable gradient computation during validation
        sys.stdout.flush()
        for data_batch in tqdm(data_loader, total=len(data_loader), desc='Testing', unit='batch', leave=False):
            sys.stderr.flush()
                
            # Extract input and labels
            X = data_batch['data'].reshape(data_batch['data'].shape[0], data_batch['data'].shape[1], data_batch['data'].shape[-1], data_batch['data'].shape[-1]).to(device=device)
            Y = data_batch['label'].to(device=device)
            Z = data_batch['segment_name']
            
            segment_names_list.append(Z)
            true_labels_list.append(torch.reshape(Y, (-1,1)))
            
            # Forward pass
            logits, predictions, prediction_proba = model(X)
            predictions_list.append(torch.reshape(predictions, (-1,1)))
            prediction_proba_list.append(torch.reshape(prediction_proba, (-1,n_classes)))
                
            # Calculate sum of total loss for epoch
            cum_loss += criterion(logits.float(), Y.long()).item() # Criterion returns a scalar tensor
            
    # Calculate loss for validation
    loss = cum_loss / len(data_loader)
    
    # Convert true label list into array
    true_labels = np.array(torch.cat(true_labels_list, dim=0).to('cpu'))
    
    # Convert the output lists into arrays and concatenate along dim=0 (rows)
    predictions = np.array(torch.cat(predictions_list, dim=0).to('cpu'))
    prediction_proba = np.array(torch.cat(prediction_proba_list, dim=0).to('cpu'))
    
    # Convert segment name list into a single list
    segment_names_list = [segment_name for batch in segment_names_list for segment_name in batch]
    
    # Print validation loss
    print('\n======> Loss: %.4f' % loss)

    # Saving
    if save:
        # pathmaster.set_file_tag(pathmaster.file_tag + '_test')
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        title = 'Evaluation Confusion Matrix'
        plot_save_func.conf_matrix(conf_matrix, title, save, pathmaster)
        
        plot_save_func.save_labels(true_labels, pathmaster)
        # plot_save_func.save_labels(np.hstack([segment_names, true_labels]), pathmaster)
        plot_save_func.save_predictions(predictions, pathmaster)
        plot_save_func.save_prediction_proba(prediction_proba, pathmaster)
        plot_save_func.metrics(true_labels, predictions, prediction_proba, save, pathmaster)
        
        plot_save_func.save_classification_report(true_labels, predictions, save, pathmaster)
        plot_save_func.save_classification_report_imbalanced(true_labels, predictions, save, pathmaster)
        
        clf_names = ['Model']
        plot_save_func.mean_roc_curves([true_labels], [prediction_proba], clf_names, save, pathmaster)
        plot_save_func.roc_curves(true_labels, prediction_proba, save, pathmaster)
        
        plot_save_func.save_segment_names(segment_names_list, pathmaster)
        plot_save_func.save_output_file(segment_names_list, predictions, true_labels, pathmaster)
        

def best_DenseNet_config_binary(data_loader, model_type=torch.float32, n_classes=2, save=False, pathmaster=None):
    print('\n===========================================================================================')
    
    # If GPU is available, use GPU, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get paths
    checkpoints_path = pathmaster.checkpoints_path()
    
    # Load model hyperparameters
    num_layers_per_dense, growth_rate, compression, bottleneck, drop_rate, _, _, _, activation = load_hyperparameters_random_search(pathmaster)
    # When testing on the test set, drop_rate, class_weights, learning_rate, and lambda_l1 are not needed
    
    # Initialize model
    model = DenseNet_config(img_channels=1, num_layers_per_dense=num_layers_per_dense, n_classes=n_classes, growth_rate=growth_rate, 
                  compression=compression, bottleneck=bottleneck, drop_rate=drop_rate,
                  activation=activation).to(device=device, dtype=model_type)
    
    # Create criterion for loss
    criterion = nn.CrossEntropyLoss()
    
    # If checkpoint is not specified, terminate the function
    checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_' + pathmaster.file_tag + '.pt')
    assert os.path.exists(checkpoint_path), 'Function terminated. Not a valid checkpoint path.'
    
    # Load model
    model = load_model(model, pathmaster)
    
    # Initialize true label lists
    true_labels_list = []
    
    # Intialize output (prediction) lists
    predictions_list = []
    prediction_proba_list = []
    
    # Initialize segment names list
    segment_names_list = []
    
    # Evaluation
    model.eval()
    cum_loss = 0
    with torch.no_grad(): # Disable gradient computation during validation
        sys.stdout.flush()
        for data_batch in tqdm(data_loader, total=len(data_loader), desc='Testing', unit='batch', leave=False):
            sys.stderr.flush()
                
            # Extract input and labels
            X = data_batch['data'].reshape(data_batch['data'].shape[0], data_batch['data'].shape[1], data_batch['data'].shape[-1], data_batch['data'].shape[-1]).to(device=device)
            Y = data_batch['label'].to(device=device)
            Z = data_batch['segment_name']
            
            segment_names_list.append(Z)
            true_labels_list.append(torch.reshape(Y, (-1,1)))
            
            # Forward pass
            logits, predictions, prediction_proba = model(X)
            predictions_list.append(torch.reshape(predictions, (-1,1)))
            prediction_proba_list.append(torch.reshape(prediction_proba, (-1,n_classes)))
                
            # Calculate sum of total loss for epoch
            cum_loss += criterion(logits.float(), Y.long()).item() # Criterion returns a scalar tensor
            
    # Calculate loss for validation
    loss = cum_loss / len(data_loader)
    
    # Convert true label list into array
    true_labels = np.array(torch.cat(true_labels_list, dim=0).to('cpu'))
    
    # Convert the output lists into arrays and concatenate along dim=0 (rows)
    predictions = np.array(torch.cat(predictions_list, dim=0).to('cpu'))
    prediction_proba = np.array(torch.cat(prediction_proba_list, dim=0).to('cpu'))
    
    # Convert segment name list into a single list
    segment_names_list = [segment_name for batch in segment_names_list for segment_name in batch]
    
    # Print validation loss
    print('\n======> Loss: %.4f' % loss)

    # Saving
    if save:
        # pathmaster.set_file_tag(pathmaster.file_tag + '_test')
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        title = 'Evaluation Confusion Matrix'
        plot_save_func.conf_matrix(conf_matrix, title, save, pathmaster, class_names=['non-AF', 'AF'])
        
        plot_save_func.save_labels(true_labels, pathmaster)
        plot_save_func.save_predictions(predictions, pathmaster)
        plot_save_func.save_prediction_proba_binary(prediction_proba, pathmaster)
        plot_save_func.metrics_binary(true_labels, predictions, prediction_proba, save, pathmaster)
        
        plot_save_func.save_classification_report(true_labels, predictions, save, pathmaster)
        plot_save_func.save_classification_report_imbalanced(true_labels, predictions, save, pathmaster)
        
        plot_save_func.roc_curves_binary(true_labels, prediction_proba, save, pathmaster, class_names=['non-AF', 'AF'])
        
        plot_save_func.save_segment_names(segment_names_list, pathmaster)
        plot_save_func.save_output_file(segment_names_list, predictions, true_labels, pathmaster)


class IdentityScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(IdentityScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Returns the current learning rate without any modifications.
        return self.base_lrs            


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path): # Will also be called to save the most recent checkpoint locally in the runtime so I always have the most recent checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else IdentityScheduler(optimizer).state_dict(),  # Create identity scheduler if missing, actually doesn't work since the parameter is required
        'epoch': epoch,
        'loss': loss
        }, checkpoint_path)
 

def save_best_checkpoint(model, optimizer, scheduler, epoch, current_loss, best_loss, pathmaster): # When training the model, best_loss should be initialized to float.('inf')
    # Might be good to have two different checkpoint paths, one for the best and one for the most recent checkpoint, maybe also have temp vs permanent checkpoint paths    
    if current_loss < best_loss:
        checkpoints_path = pathmaster.checkpoints_path()
        checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_' + pathmaster.file_tag + '.pt')
        best_loss = current_loss
        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, checkpoint_path)
        print('\nNew checkpoint with better loss was saved!')
        
        return best_loss
    else:
        return best_loss


def load_model(model, pathmaster):
    checkpoints_path = pathmaster.checkpoints_path()
    checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_' + pathmaster.file_tag + '.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print('\nModel loaded!')
        # print(f'Resuming training from epoch {start_epoch}, batch {start_batch}')
        
        return model
    else:
        print('\nError! Model does not exist!')


def load_hyperparameters_random_search(pathmaster):
    hyperparameters_path = pathmaster.hyperparameters_path()
    
    # Extract model hyperparameters
    model_hyperparameters_file = os.path.join(hyperparameters_path, 'hyperparameters_' + pathmaster.file_tag + '.csv')
    model_hyperparameters = pd.read_csv(model_hyperparameters_file)
    num_layers_per_dense = int(model_hyperparameters['num_layers_per_dense'].iloc[0])
    growth_rate = int(model_hyperparameters['growth_rate'].iloc[0])
    compression = model_hyperparameters['compression'].iloc[0]
    bottleneck = model_hyperparameters['bottleneck'].iloc[0]
    drop_rate = model_hyperparameters['drop_rate'].iloc[0]
    class_weights = model_hyperparameters['class_weights']
    learning_rate = model_hyperparameters['learning_rate'].iloc[0]
    lambda_l1 = model_hyperparameters['lambda_l1'].iloc[0]
    activation = string_to_activation((model_hyperparameters['activation'].iloc[0]))
    
    return num_layers_per_dense, growth_rate, compression, bottleneck, drop_rate, class_weights, learning_rate, lambda_l1, activation


def string_to_activation(activation_string):
    activation_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(),
        'softplus': nn.Softplus(),
        'softshrink': nn.Softshrink(),
        'softmin': nn.Softmin(),
        'log_softmax': nn.LogSoftmax(),
        'elu': nn.ELU(),
        'prelu': nn.PReLU(),
        'relu6': nn.ReLU6(),
        'rrelu': nn.RReLU(),
        'celu': nn.CELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        # Add more activation functions if needed
    }
    
    return activation_map.get(activation_string, None)


def activation_to_string(activation_func):
    activation_map = {
        nn.ReLU: 'relu',
        nn.LeakyReLU: 'leaky_relu',
        nn.Sigmoid: 'sigmoid',
        nn.Tanh: 'tanh',
        nn.Softmax: 'softmax',
        nn.Softplus: 'softplus',
        nn.Softshrink: 'softshrink',
        nn.Softmin: 'softmin',
        nn.LogSoftmax: 'log_softmax',
        nn.ELU: 'elu',
        nn.PReLU: 'prelu',
        nn.ReLU6: 'relu6',
        nn.RReLU: 'rrelu',
        nn.CELU: 'celu',
        nn.SELU: 'selu',
        nn.GELU: 'gelu',
        nn.SiLU: 'silu',
        # Add more activation functions if needed
    }
    
    return activation_map.get(activation_func.__class__, 'unknown')


class EarlyStoppingCallback:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, epoch, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}. No improvement for {self.patience} epochs.")
                
                return True
        
        return False