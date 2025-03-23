# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:29:59 2024

@author: dchen
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import ToTensor
from numpy import random
import cv2
from pyarrow import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class CustomDataset(Dataset):
    def __init__(self, combination_path, groups, standardize=True, data_format='pt', start_idx=0, 
                 img_channels=1, img_size=128, downsample=None, data_type=torch.float32, is_tfs=True, binary=False):
        self.combination_path = combination_path
        self.standardize = standardize
        self.data_format = data_format
        self.transforms = ToTensor()
        self.start_idx = start_idx  # Initial batch index to start from, useful for resuming training
        self.img_channels = img_channels
        self.img_size = img_size
        self.downsample = downsample
        self.is_tfs = is_tfs
        self.groups = groups
        self.dtype = data_type
        self.binary = binary
        
        self.refresh_dataset()


    def refresh_dataset(self):
        self.segment_names, self.labels = self.extract_segment_names_and_labels()


    def __len__(self): # Method is implicitly called when len() is used on an instance of CustomDataset
        return len(self.segment_names)


    def __getitem__(self, idx): # Method is implicitly called when getitem() is used on an instance of CustomDataset. It is called batch_size number of times per iteration of dataloader | Loads segments as needed (lazy loading)
        actual_idx = (idx + self.start_idx) % len(self.segment_names)  # Adjust index based on start_idx and wrap around if needed (i.e. index falls out of bounds)
        segment_name = self.segment_names[actual_idx]
        label = self.labels[segment_name]

        data_tensor = self.load_data(segment_name)

        return {'data': data_tensor, 'label': label, 'segment_name': segment_name}
    
        # When iterating over the dataloader, which returns batches of data, each batch will contain a dictionary with keys corresponding to the data and labels.

        # Since the dataloader's dataset's __getitem__ method returns a dictionary with keys 'data', 'label', and 'segment_name', the returned batch will be a dictionary where:

        # The 'data' key will correspond to a tensor of shape (batch_size, ...), representing the shape of the data.
        # The 'label' key will correspond to a tensor of shape (batch_size, ...), representing the shape of the labels.
        # The 'segment_name' key will correspond to a tensor of shape (batch_size, ...), representing the shape of the segment_name.

    def set_start_idx(self, index):
        self.start_idx = index
      
            
    def extract_segment_names_and_labels(self): # Only extract the segments and labels of a particular class, temporary solution
        segment_names = []
        labels = {}
        
        group_directories = [entry for entry in os.listdir(self.combination_path) if os.path.isdir(os.path.join(self.combination_path, entry))]
        group = list(set(self.groups).intersection(set(group_directories)))[0]
        
        combination = self.second_to_last_directory_name(self.combination_path)
        label_file = os.path.join(self.combination_path, combination + '_' + group + '_names_labels.csv')
        if os.path.exists(label_file):                
            # Use PyArrow to read csv
            parse_options = csv.ParseOptions(delimiter=',') # Indicate delimiter
            read_options = csv.ReadOptions(column_names=['segment_names', 'labels'], skip_rows=1) # Assign desired column names and skip the first row (headers)
            label_data = csv.read_csv(label_file, parse_options=parse_options, read_options=read_options)
            label_data = label_data.to_pandas()
                
            label_segment_names = label_data['segment_names']
            for idx, segment_name in enumerate(label_segment_names): # enumerate() returns the value and corresponding index of each element in an iterable
                label_val = label_data['labels'].values[idx]
                segment_names.append(segment_name)
                
                if self.binary and label_val == 2:
                    labels[segment_name] = 0
                else:
                    labels[segment_name] = label_val

        return segment_names, labels
    
    
    def second_to_last_directory_name(self, path):
        # Normalize path separator to '/'
        path = path.replace('\\', '/')

        # Split the path into its components
        components = path.split('/')

        # Remove empty components
        components = [c for c in components if c]

        # Check if the path ends with a separator (indicating it's a directory)
        if path.endswith('/'):
            # Remove the last empty component
            components.pop()

        # If there's only one or zero directories in the path, return None
        if len(components) <= 1:
            return None

        # Return the name of the second-to-last directory
        return components[-2]


    def load_data(self, segment_name):
        data_path_group = os.path.join(self.combination_path, segment_name.split('_')[1])
        seg_path = os.path.join(data_path_group, segment_name + '.' + self.data_format)
            
        try: # Allows to define a block of code to be executed and specify how to handle any errors that might occur during its execution
            if self.data_format == 'csv' and seg_path.endswith('.csv'):
                # data_plot = np.array(pd.read_csv(seg_path, header=None))
                
                # Use PyArrow to read csv
                read_options = csv.ReadOptions(autogenerate_column_names=True)
                seg_data = csv.read_csv(seg_path, read_options=read_options)
                data_plot = seg_data.to_pandas().to_numpy()
                
                data_tensor = torch.tensor(data_plot).reshape(self.img_channels, self.img_size, self.img_size)
            elif self.data_format == 'png' and seg_path.endswith('.png'):
                img = Image.open(seg_path)
                img_data = np.array(img)
                data_tensor = torch.tensor(img_data).unsqueeze(0)
            elif self.data_format == 'pt' and seg_path.endswith('.pt'):
                data_tensor = torch.load(seg_path)
            else:
                raise ValueError("Unsupported file format")

            if self.downsample is not None:
                # Downsample the image
                # Use OpenCV to resize the array to downsample x downsample using INTER_AREA interpolation
                data_array = cv2.resize(np.array(data_tensor.reshape(self.img_size, self.img_size).to('cpu')), (self.downsample, self.downsample), interpolation=cv2.INTER_AREA)
                data_tensor = torch.tensor(data_array, dtype=self.dtype).reshape(self.img_channels, self.downsample, self.downsample)
            else:
                data_tensor = data_tensor.reshape(self.img_channels, self.img_size, self.img_size).to(self.dtype)

            if self.standardize:
                data_tensor = self.standard_scaling(data_tensor) # Standardize the data
            
            return data_tensor

        except Exception as e:
            print(f"Error processing segment: {segment_name}. Exception: {str(e)}")
            if self.downsample is not None:
                return torch.zeros((self.img_channels, self.downsample, self.downsample))  # Return zeros in case of an error
            else:
                return torch.zeros((self.img_channels, self.img_size, self.img_size))  # Return zeros in case of an error

    def standard_scaling(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape) # Converts data into 2D array, standardizes it, reshapes it back into 3D (1,X,X)
        return torch.tensor(data, dtype=self.dtype)

def load_data_split_batched(combination_path, groups, batch_size, standardize=False, data_format='csv', 
                            drop_last=False, num_workers=4, start_idx=0, 
                            img_channels=1, img_size=128, downsample=None, data_type=torch.float32, is_tfs=True, binary=False):
    torch.manual_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
    
    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True
    
    dataset = CustomDataset(combination_path, groups, standardize, data_format, start_idx=start_idx, 
                            img_channels=img_channels, img_size=img_size, downsample=downsample, data_type=data_type, is_tfs=is_tfs, binary=binary)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g) # Prefetches 2 batches ahead of current training iteration (allows loading of data simultaneously with training). Shuffle is set to False to resume training at a specific batch.
    return dataloader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Function to extract and preprocess data
def preprocess_data(combination, split, batch_size, standardize=False, img_channels=1, img_size=128, 
                    downsample=None, data_type=torch.float32, pathmaster=None, binary=False):
    start_idx = 0
    combination_path = pathmaster.combination_path(combination, split)
    data_format = 'pt'
    
    num_workers = 8
    
    train_loader = load_data_split_batched(combination_path, ['train'], batch_size, standardize=standardize, 
                                           data_format=data_format, num_workers=num_workers,
                                           start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                           data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary)
    val_loader = load_data_split_batched(combination_path, ['validate'], batch_size, standardize=standardize, 
                                         data_format=data_format, num_workers=num_workers, 
                                         start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                         data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary)

    return train_loader, val_loader
        
    