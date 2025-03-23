import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np
import time
import datetime as dt
import os
import sys

# Import my own functions and classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pathmaster import PathMaster
from utils import plot_save_func
import utils.model_func as model_func
import utils.dataloader as dataloader
import utils.dataloader_combined as dataloader_combined
from models.densenet_configurable import DenseNet

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def load_data(binary, dataset, pathmaster):
    # Image dimensions
    img_channels = 1
    img_size = 128 
    downsample = None
    standardize = True
    
    # Data loading details
    data_format = 'pt'
    batch_size = 256
    
    # Data type: the type to convert the data into when it is loaded in
    data_type = torch.float32
    
    if dataset == 'pulsewatch' or dataset == 'Pulsewatch':
        # Split UIDs
        train_set, val_set, test_set = dataloader.split_uids_60_10_30_smote(pathmaster)
        # train_set, val_set, test_set = dataloader.split_uids_60_10_30_v2(pathmaster)
        # train_set, val_set, test_set = dataloader.split_uids_60_10_30_noNSR(pathmaster)
        
        # Data loaders
        train_loader, val_loader, test_loader = dataloader.preprocess_data(data_format, train_set, val_set, test_set, 
                                                    batch_size, standardize, False, img_channels, img_size, downsample, data_type, pathmaster, binary)
    elif dataset == 'combined' or dataset == 'Combined':
        train_loader, val_loader = dataloader_combined.preprocess_data('combined_dataset', 'multiclass', batch_size, standardize, img_channels, img_size,
                                                                       downsample, data_type, pathmaster, binary)
        # Split UIDs
        train_set, val_set, test_set = dataloader.split_uids_60_10_30_smote(pathmaster)
        _, _, test_loader = dataloader.preprocess_data(data_format, train_set, val_set, test_set, 
                                                       batch_size, standardize, False, img_channels, img_size, downsample, data_type, pathmaster, binary)
    else:
        print('Invalid dataset')
        train_loader, val_loader, test_loader = [], [], []
        
    return train_loader, val_loader, test_loader


def train_DenseNet(train_loader, val_loader, config, pathmaster, n_epochs, n_classes, save=False):
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
    
    model = DenseNet(img_channels, config['num_layers_per_dense'], n_classes, config['growth_rate'], config['compression'], 
                     config['bottleneck'], config['drop_rate'], config['activation']).to(device=device)
    
    # Loss function and optimizer
    criterion_train = nn.CrossEntropyLoss(weight=torch.tensor(config['class_weights']).to(device=device))
    criterion_val = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = model_func.IdentityScheduler(optimizer)
        
    
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
    early_stopping_callback = model_func.EarlyStoppingCallback(round(n_epochs / 10) if n_epochs > 50 else 5)
    
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
            # Extract input and labels: train_batch['data'].shape = [batch_size, img_channels, img_size, img_size]
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
        best_loss_val = model_func.save_best_checkpoint(model, optimizer, scheduler, epoch, epoch_loss_val, best_loss_val, pathmaster)
        
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
        
    print("Finished iteration!")
    

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


def best_DenseNet(data_loader, model, n_classes=3, save=False, pathmaster=None):
    # If GPU is available, use GPU, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create criterion for loss
    criterion = nn.CrossEntropyLoss()
    
    # Initialize true label lists
    true_labels_list = []
    
    # Intialize output (prediction) lists
    predictions_list = []
    prediction_proba_list = []
    
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
    
    # Print validation loss
    print('\n======> Loss: %.4f' % loss)

    # Saving
    if save:
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        title = 'Evaluation Confusion Matrix'
        plot_save_func.conf_matrix(conf_matrix, title, save, pathmaster)
        
        plot_save_func.save_labels(true_labels, pathmaster)
        plot_save_func.save_predictions(predictions, pathmaster)
        plot_save_func.save_prediction_proba(prediction_proba, pathmaster)
        plot_save_func.metrics(true_labels, predictions, prediction_proba, save, pathmaster)
        
        plot_save_func.save_classification_report(true_labels, predictions, save, pathmaster)
        plot_save_func.save_classification_report_imbalanced(true_labels, predictions, save, pathmaster)
        
        clf_names = ['Model']
        plot_save_func.mean_roc_curves([true_labels], [prediction_proba], clf_names, save, pathmaster)
        plot_save_func.roc_curves(true_labels, prediction_proba, save, pathmaster)


def best_DenseNet_binary(data_loader, model, n_classes=3, save=False, pathmaster=None):
    # If GPU is available, use GPU, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create criterion for loss
    criterion = nn.CrossEntropyLoss()
    
    # Initialize true label lists
    true_labels_list = []
    
    # Intialize output (prediction) lists
    predictions_list = []
    prediction_proba_list = []
    
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
        
        plot_save_func.roc_curves(true_labels, prediction_proba, save, pathmaster, class_names=['non-AF', 'AF'])


def main(n_iter=100, n_epochs=50, save=False, start_iter=0):
    # Device and drives
    is_linux = False
    is_hpc = False
    is_internal = False
    is_external = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    binary = False
    
    # Input
    is_tfs = True
    
    # Intialize the focus
    focus = 'random_search'
    
    # Initialize the file tag
    file_tag = 'does not even matter lol'
    
    # Image resolution
    # img_res = '256x256'
    img_res = '128x128_float16'
    
    # Other parameters
    if binary:
        n_classes = 2
    else:
        n_classes = 3
    img_channels = 1
    
    # Create a PathMaster object
    pathmaster = PathMaster(is_linux, is_hpc, is_tfs, is_internal, is_external, focus, file_tag, img_res)
    
    # Config
    config = { # Default, no bottleneck or compression
        'num_layers_per_dense': [2, 4, 6, 8, 10, 12, 14],
        'growth_rate': [12, 16, 20, 24, 28, 32],
        'compression': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        'bottleneck': [True, False],
        'drop_rate': [0.0, 0.1, 0.2, 0.3, 0.4],
        'class_weights': [[61006/39122, 61006/13993, 61006/7891]], # [[59743/(38082+7861), 59743/13800]],
        'learning_rate': np.logspace(-5, -2, 10),
        'lambda_l1': np.logspace(-5, -1, 10),
        'activation': [nn.ReLU(), nn.ELU(), nn.SiLU(), nn.LeakyReLU(), nn.GELU(), nn.CELU(), nn.PReLU()],
        }
    
    # Load data
    dataset = 'pulsewatch'
    train_loader, val_loader, test_loader = load_data(binary, dataset, pathmaster)
    
    for iter in tqdm(range(n_iter), total=n_iter, desc='Optimization', unit='Iteration', leave=False):
        # Hyperparameters
        hyperparameters = {}
        for key, values in config.items():
            hyperparameters[key] = random.choice(values)
        
        # Resuming random search
        if iter < start_iter:
            print('\nSkipped Iteration', iter, '\n')
            continue
        
        # Skip if hyperparameters results in too complex model
        if (hyperparameters['num_layers_per_dense'] >= 10 and hyperparameters['growth_rate'] >= 24 and hyperparameters['compression'] >= 0.8) or (hyperparameters['num_layers_per_dense'] >= 8 and hyperparameters['growth_rate'] >= 28 and hyperparameters['compression'] == 1):
            print('\nModel too complex. Skipped Iteration', iter, '\n')
            continue
        
        # Set file tag
        pathmaster.set_file_tag('random_search_' + str(iter))
        # pathmaster.set_file_tag('random_search_' + str(iter) + '_v2')
        
        # Train model
        train_DenseNet(train_loader, val_loader, hyperparameters, pathmaster, n_epochs, n_classes, save=True)
        
        # Load model
        model = DenseNet(img_channels, hyperparameters['num_layers_per_dense'], n_classes, hyperparameters['growth_rate'], hyperparameters['compression'], 
                     hyperparameters['bottleneck'], hyperparameters['drop_rate'], hyperparameters['activation']).to(device=device)
        model = model_func.load_model(model, pathmaster)
        
        # Test model
        if not binary:
            # best_DenseNet(val_loader, model, n_classes, save, pathmaster)
            # pathmaster.set_file_tag(pathmaster.file_tag + '_test')
            best_DenseNet(test_loader, model, n_classes, save, pathmaster)
        else:
            # best_DenseNet_binary(val_loader, model, n_classes, save, pathmaster)
            # pathmaster.set_file_tag(pathmaster.file_tag + '_test')
            best_DenseNet_binary(test_loader, model, n_classes, save, pathmaster)
        
        
if __name__ == "__main__":
    main(n_iter=100, n_epochs=100, save=True, start_iter=0)