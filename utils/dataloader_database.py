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
    def __init__(self, data_path, labels_path, standardize=True, data_format='pt', start_idx=0, 
                 img_channels=1, img_size=128, downsample=None, data_type=torch.float32, is_tfs=True, binary=False, uids=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.standardize = standardize
        self.data_format = data_format
        self.transforms = ToTensor()
        self.start_idx = start_idx  # Initial batch index to start from, useful for resuming training
        self.img_channels = img_channels
        self.img_size = img_size
        self.downsample = downsample
        self.is_tfs = is_tfs
        self.dtype = data_type
        self.binary = binary
        self.uids = uids
        
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
    
    
    def filter_by_prefix(self, text_list, prefixes):
        """
        This function filters a list of strings based on prefixes from another list.

        Args:
            text_list: A list of strings to be filtered.
            prefixes: A list of strings representing the prefixes to filter by.

        Returns:
            A new list containing only the strings from text_list that start with any prefix in prefixes.
        """
        return [text for text in text_list if any(text.startswith(prefix) for prefix in prefixes)]
    
    
    def extract_segment_names_and_labels(self):
        segment_names = []
        labels = {}
        label_file = self.labels_path
        if os.path.exists(label_file):                
            # Use PyArrow to read csv
            parse_options = csv.ParseOptions(delimiter=',') # Indicate delimiter
            read_options = csv.ReadOptions(column_names=['segment_names', 'labels'], skip_rows=1) # Assign desired column names and skip the first row (headers)
            label_data = csv.read_csv(label_file, parse_options=parse_options, read_options=read_options)
            label_data = label_data.to_pandas()
                
            label_segment_names = label_data['segment_names']
            
            if self.uids is not None:
                # Filter segment names based on whether they start with any string within the filter_strings list
                label_segment_names = self.filter_by_prefix(label_segment_names, self.uids)\
                    
                # Filtering the DataFrame
                label_data = label_data[label_data['segment_names'].isin(label_segment_names)]
                label_data.reset_index(drop=True, inplace=True)
            
            for idx, segment_name in enumerate(label_segment_names): # enumerate() returns the value and corresponding index of each element in an iterable
                label_val = label_data['labels'].values[idx]
                
                if self.binary and label_val == 2: # If binary is true, set all PAC/PVC to 0 (non-AF)
                    label_val = 0
                
                segment_names.append(segment_name)
                labels[segment_name] = label_val
        else:
            print('Missing labels file:', label_file)

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
        seg_path = os.path.join(self.data_path, segment_name + '.' + self.data_format)
            
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

def load_data_split_batched(data_path, labels_path, batch_size, standardize=False, data_format='csv', 
                            drop_last=False, num_workers=4, start_idx=0, 
                            img_channels=1, img_size=128, downsample=None, data_type=torch.float16, is_tfs=True, binary=False, uids=None):
    torch.manual_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
    
    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True
    
    dataset = CustomDataset(data_path, labels_path, standardize, data_format, start_idx=start_idx, 
                            img_channels=img_channels, img_size=img_size, downsample=downsample, data_type=data_type, is_tfs=is_tfs, binary=binary, uids=uids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g) # Prefetches 2 batches ahead of current training iteration (allows loading of data simultaneously with training). Shuffle is set to False to resume training at a specific batch.
    return dataloader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def split_uids(database, test=True):
    if database == 'MIMICIII' or database == 'mimiciii' or database == 'mimicIII' or database == 'mimic3':
        if test:
            # 60:10:30 split
            train_set = ['NSR_with_PPG_1', 'NSR_with_PPG_4', 'AF_with_PPG_5', 'PAC_with_PPG_1']
            val_set = ['NSR_with_PPG_5', 'PAC_with_PPG_3']
            test_set = ['NSR_with_PPG_2', 'NSR_with_PPG_3', 'AF_with_PPG_2', 'PAC_with_PPG_2']
        else:
            # Train and validate split, 60:40 split (combine validation and test sets together)
            train_set = ['NSR_with_PPG_1', 'NSR_with_PPG_4', 'AF_with_PPG_5', 'PAC_with_PPG_1']
            val_set = ['NSR_with_PPG_5', 'PAC_with_PPG_3', 'NSR_with_PPG_2', 'NSR_with_PPG_3', 'AF_with_PPG_2', 'PAC_with_PPG_2']
            test_set = []
            
        return train_set, val_set, test_set
    elif database == 'Simband' or database == 'simband':
        if test:
            # 60:10:30 split
            train_set = ['4005', '4008', '4026', '4029', '4030', '4032', '4033', '4035', '4036', '4040', '4041', '4042',
                        '4006', '4007', '4012', '4016', '4038',
                        '4017', '4021', '4022']
            val_set = ['4002', '4028',
                    '4043',
                    '4037']
            test_set = ['4019', '4020', '4025', '4027', '4044', '4045',
                        '4013', '4015', '4024',
                        '4001', '4034']
            # test_set = train_set + val_set + test_set
        else:    
            # Train and validate split, 60:40 split (combine validation and test sets together)
            train_set = ['4005', '4008', '4026', '4029', '4030', '4032', '4033', '4035', '4036', '4040', '4041', '4042',
                        '4006', '4007', '4012', '4016', '4038',
                        '4017', '4021', '4022']
            val_set = ['4002', '4028',
                    '4043',
                    '4037',
                    '4019', '4020', '4025', '4027', '4044', '4045',
                    '4013', '4015', '4024',
                    '4001', '4034']
            test_set = []
        
        return train_set, val_set, test_set
    else:
        print('Invalid Database')


# Function to extract and preprocess data
def preprocess_data(database, batch_size, standardize=False, img_channels=1, img_size=128, 
                    downsample=None, data_type=torch.float32, pathmaster=None, binary=False):
    start_idx = 0
    
    if database == 'DeepBeat' or database == 'deepbeat' or database == 'Deepbeat':
        data_path, labels_path = pathmaster.deepbeat_paths()
    elif database == 'MIMICIII' or database == 'mimiciii' or database == 'mimicIII' or database == 'mimic3':
        data_path, labels_path = pathmaster.mimic3_paths()
    elif database == 'Simband' or database == 'simband':
        data_path, labels_path = pathmaster.simband_paths()
    else:
        print('Invalid Database')
    
    data_format = 'pt'
    
    num_workers = 1
    
    test_loader = load_data_split_batched(data_path, labels_path, batch_size, standardize=standardize, 
                                           data_format=data_format, num_workers=num_workers,
                                           start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                           data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary)

    return test_loader
    
def preprocess_data_split(database, batch_size, standardize=False, img_channels=1, img_size=128, 
                    downsample=None, data_type=torch.float32, pathmaster=None, binary=False, test=True):
    start_idx = 0
    
    if database == 'MIMICIII' or database == 'mimiciii' or database == 'mimicIII' or database == 'mimic3':
        data_path, labels_path = pathmaster.mimic3_paths()
    elif database == 'Simband' or database == 'simband':
        data_path, labels_path = pathmaster.simband_paths()
    else:
        print('Invalid Database')
        
    labels_path = labels_path[:-4] + '_subject_independent.csv'
    
    train_set, val_set, test_set = split_uids(database, test)
    
    data_format = 'pt'
    
    num_workers = 1
    
    train_loader = load_data_split_batched(data_path, labels_path, batch_size, standardize=standardize, 
                                           data_format=data_format, num_workers=num_workers,
                                           start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                           data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary, uids=train_set)
    val_loader = load_data_split_batched(data_path, labels_path, batch_size, standardize=standardize, 
                                           data_format=data_format, num_workers=num_workers,
                                           start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                           data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary, uids=val_set)
    if test:
        test_loader = load_data_split_batched(data_path, labels_path, batch_size, standardize=standardize, 
                                            data_format=data_format, num_workers=num_workers,
                                            start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                            data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary, uids=test_set)
    else:
        test_loader = []

    return train_loader, val_loader, test_loader