import torch
import os
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sys

# Import my own functions and classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.pathmaster import PathMaster
from utils import dataloader

def apply_smote(data, labels):
    smote = SMOTE(random_state=42,sampling_strategy='not majority',k_neighbors=5)
    data_resampled, labels_resampled = smote.fit_resample(data, labels)
    return data_resampled, labels_resampled

def apply_borderline_smote(data, labels):
    borderline_smote = BorderlineSMOTE(random_state=42,sampling_strategy='not majority',k_neighbors=5)
    data_resampled, labels_resampled = borderline_smote.fit_resample(data, labels)
    return data_resampled, labels_resampled

def apply_adasyn(data, labels):
    adasyn = ADASYN(random_state=42,sampling_strategy='not majority',n_neighbors=5)
    data_resampled, labels_resampled = adasyn.fit_resample(data, labels)
    return data_resampled, labels_resampled

def save_image(i, image, group, save_dir):
    
    # Generate a unique file name with zero-padding
    file_name = f'{i+1:06d}' + '_' + group + '_tfs'
    
    # Convert the image to a PyTorch tensor
    tensor_image = torch.tensor(image).to(dtype=torch.float16)
    tensor_image = tensor_image.reshape(tensor_image.size()[-2], tensor_image.size()[-2])

    
    # Save the tensor to a .pt file
    torch.save(tensor_image, os.path.join(save_dir, file_name + '.pt'))
    
    return file_name

def save_images_parallel(data_resampled, group, save_dir):
    file_names = []
    with ProcessPoolExecutor() as executor:
        results = [executor.submit(save_image, i, image, group, save_dir) for i, image in enumerate(data_resampled)]
        for future in results:
            file_names.append(future.result())
    return file_names

def main():
    # Initialize save location specifics
    smote_type = 'SMOTE'
    # smote_type = 'Borderline_SMOTE'
    # smote_type = 'ADASYN'
    
    split = 'holdout_60_10_30'
    
    groups = ['train', 'validate', 'test']
    
    # Device and drives
    is_linux = False
    is_hpc = False
    is_internal = False
    is_external = False
        
    # Input
    is_tfs = True
        
    # Image resolution
    img_res = '128x128_float16'
        
    # Data type: the type to convert the data into when it is loaded in
    data_type = torch.float32
        
    # Create a PathMaster object
    pathmaster = PathMaster(is_linux, is_hpc, is_tfs, is_internal, is_external, '_', '_', img_res)
        
    # Image dimensions
    img_channels = 1
    img_size = 128 
    downsample = None
    standardize = None
        
    # Split UIDs
    train_set, val_set, test_set = dataloader.split_uids_60_10_30_smote(pathmaster)
        
    # Preprocess data
    data_format = 'pt'
    batch_size = 256
    
    train_loader, val_loader, test_loader = dataloader.preprocess_data(data_format, train_set, val_set, test_set, 
                                                      batch_size, standardize, False, img_channels, img_size, downsample, data_type, pathmaster)
    data_loaders = [train_loader, val_loader, test_loader]
    print()
    sys.stdout.flush()
    for data_loader, group in tqdm(zip(data_loaders,groups), total=len(data_loaders), desc='SMOTE', unit='Data Loader', leave=False):    
        sys.stderr.flush()
        
        # Define your original data and labels
        data = np.empty((0,img_size*img_size))
        labels = np.empty((0,1))
        
        sys.stdout.flush()
        
        for data_batch in tqdm(data_loader, total=len(data_loader), desc='Loading', unit='batch', leave=False):
            sys.stderr.flush()
            
            # Extract input and labels
            X = data_batch['data'].reshape(data_batch['data'].shape[0], data_batch['data'].shape[-1] * data_batch['data'].shape[-1]).numpy()
            Y = data_batch['label'].numpy().reshape(-1,1)
            
            data = np.concatenate((data, X), axis=0)
            labels = np.concatenate((labels, Y), axis=0)
        
        sys.stderr.flush()
        print('\nData shape:', data.shape)
        print('Labels shape:', labels.shape)
        sys.stdout.flush()
        
        if group != 'test':
            # SMOTE
            if smote_type == 'SMOTE':
                data_resampled, labels_resampled = apply_smote(data, labels)
            elif smote_type == 'Borderline_SMOTE':
                data_resampled, labels_resampled = apply_borderline_smote(data, labels)
            elif smote_type == 'ADASYN':
                data_resampled, labels_resampled = apply_adasyn(data, labels)
            else:
                raise ValueError('Not a valid SMOTE type')
            data_resampled = data_resampled.reshape(len(data_resampled), img_channels, img_size, img_size)
            sys.stderr.flush()
            print('\nResampled Data shape:', data_resampled.shape)
            print('Resampled Labels shape:', labels_resampled.shape)
            print()
            sys.stdout.flush()
        else:
            data_resampled = data
            data_resampled = data_resampled.reshape(len(data_resampled), img_channels, img_size, img_size)
            labels_resampled = labels
            sys.stderr.flush()
            print('\nResampled Data shape:', data_resampled.shape)
            print('Resampled Labels shape:', labels_resampled.shape)
            print()
            sys.stdout.flush()
        
        # Define a directory to save the images
        save_dir = os.path.join(os.path.dirname(pathmaster.data_path), smote_type, split, group)
        os.makedirs(save_dir, exist_ok=True)
        
        file_names = save_images_parallel(data_resampled, group, save_dir)
            
        # Ground truths
        data_labels = pd.DataFrame({
            'segment_name': file_names,
            'label': labels_resampled.reshape(-1)
        })
        csv_file_name = os.path.join(os.path.dirname(pathmaster.data_path), smote_type, split, smote_type + '_' + group + '_names_labels.csv')
        data_labels.to_csv(csv_file_name, index=False)

if __name__ == '__main__':
    main()