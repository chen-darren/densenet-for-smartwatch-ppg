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
import math
from numpy import random
from numpy.random import choice
import cv2
from pyarrow import csv
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def split_uids_60_10_30(pathmaster):
    # ====== Load the per subject arrythmia summary ======
    file_path = pathmaster.summary_path()
    # df_summary = pd.read_csv(file_path)

    # Read the CSV file using pyarrow.csv.read_csv
    table_summary = csv.read_csv(file_path)
    df_summary = table_summary.to_pandas()
    
    df_summary['UID'] = df_summary['UID'].astype(str).str.zfill(3) # Pads each UIDs with enough zeroes to be 3 characters
    
    df_summary['sample_nonAF'] = df_summary['NSR'] + df_summary['PACPVC'] + df_summary['SVT']
    df_summary['sample_AF'] = df_summary['AF']
    
    df_summary['sample_nonAF_ratio'] = df_summary['sample_nonAF'] / (df_summary['sample_AF'] + df_summary['sample_nonAF'])
    
    # Filter out 0-segment UIDs and UIDs without NSR, AF, and/or PAC/PVC
    remaining_UIDs = []
    
    for index, row in df_summary.iterrows():
        UID = row['UID']
        if row['TOTAL'] == 0:
            # There is no segment in this subject, skip this UID.
            print(f'---------UID {UID} has no segments.------------')
        elif (row['NSR'] > 0 or row['AF'] > 0 or row['PACPVC'] > 0): # Append UID only if it contains NSR, AF, or PAC/PVC
            remaining_UIDs.append(UID)
        else:
            print(f'---------UID {UID} has no AF, NSR, or PAC/PVC segments.------------')
        
    # Split UIDs    
    uid_nsr_train = ['011', '014', '030', '037', '044', '050', '055', '058', '074', '083', '091', '098', '101', '106', '109', '119']
    uid_nsr_val = ['041', '056', '325']
    uid_nsr_test = ['003', '012', '020', '024', '027', '035', '036', '047']
    
    uid_af_train = ['017', '301', '302', '305', '306', '318', '319', '320', '321', '322', '324', '329', '402', '405', '406', '407', '416', '420', '421']
    uid_af_val = ['400', '409', '422']
    uid_af_test = ['307', '310', '311', '312', '410', '413', '414', '415', '423']
    
    uid_pacpvc_train = ['005', '007', '013', '021', '022', '026', '028', '029', '042', '064', '068', '073', '080', '086', '087', '089', '093', '104', '110', '113', '120', '327', '408']
    uid_pacpvc_val = ['045', '054', '112']
    uid_pacpvc_test = ['002', '038', '039', '052', '053', '069', '070', '075', '078', '090', '100', '419']
    
    # Total UID counts
    total_uid_nsr = uid_nsr_train + uid_nsr_val + uid_nsr_test
    total_uid_af = uid_af_train + uid_af_val + uid_af_test
    total_uid_pacpvc = uid_pacpvc_train + uid_pacpvc_val + uid_pacpvc_test
    total_uid = total_uid_pacpvc + total_uid_af + total_uid_nsr
    
    print('Number of total and unique UIDs:', len(total_uid),'|', len(np.unique(total_uid)))
    print('Number of total and unique NSR UIDs:', len(total_uid_nsr),'|', len(np.unique(total_uid_nsr)))
    print('Number of total and unique AF UIDs:', len(total_uid_af),'|', len(np.unique(total_uid_af)))
    print('Number of total and unique PAC/PVC UIDs:', len(total_uid_pacpvc),'|', len(np.unique(total_uid_pacpvc)))
    
    train_set = uid_nsr_train + uid_af_train + uid_pacpvc_train
    val_set = uid_nsr_val + uid_af_val + uid_pacpvc_val
    test_set = uid_nsr_test + uid_af_test + uid_pacpvc_test
    
    # Limit data set size to reduce computational load for optimization
    test_set = test_set
    
    return train_set, val_set, test_set
    

def split_uids_60_10_30_smote(pathmaster):
    # ====== Load the per subject arrythmia summary ======
    file_path = pathmaster.summary_path()
    # df_summary = pd.read_csv(file_path)

    # Read the CSV file using pyarrow.csv.read_csv
    table_summary = csv.read_csv(file_path)
    df_summary = table_summary.to_pandas()
    
    df_summary['UID'] = df_summary['UID'].astype(str).str.zfill(3) # Pads each UIDs with enough zeroes to be 3 characters
    
    df_summary['sample_nonAF'] = df_summary['NSR'] + df_summary['PACPVC'] + df_summary['SVT']
    df_summary['sample_AF'] = df_summary['AF']
    
    df_summary['sample_nonAF_ratio'] = df_summary['sample_nonAF'] / (df_summary['sample_AF'] + df_summary['sample_nonAF'])
    
    # Filter out 0-segment UIDs and UIDs without NSR, AF, and/or PAC/PVC
    remaining_UIDs = []
    
    for index, row in df_summary.iterrows():
        UID = row['UID']
        if row['TOTAL'] == 0:
            # There is no segment in this subject, skip this UID.
            print(f'---------UID {UID} has no segments.------------')
        elif (row['NSR'] > 0 or row['AF'] > 0 or row['PACPVC'] > 0): # Append UID only if it contains NSR, AF, or PAC/PVC
            remaining_UIDs.append(UID)
        else:
            print(f'---------UID {UID} has no AF, NSR, or PAC/PVC segments.------------')
        
    # Split UIDs    
    uid_nsr_train = ['003', '020', '024', '041', '044', '047', '049', '050', '058', '063', '077', '084', '088', '091', '098', '099', '106', '109', '111', '118', '325']
    uid_nsr_val = ['014', '030', '036', '074']
    uid_nsr_test = ['011', '012', '027', '035', '037', '055', '056', '057', '083', '094', '101', '119']
    
    uid_af_train = ['017', '302', '306', '307', '310', '311', '319', '321', '324', '400', '402', '405', '406', '407', '409', '410', '415', '420', '421']
    uid_af_val = ['416', '422', '423']
    uid_af_test = ['301', '305', '312', '318', '320', '322', '329', '413', '414']
    
    uid_pacpvc_train = ['005', '007', '013', '021', '022', '026', '028', '029', '042', '064', '068', '073', '080', '086', '087', '089', '093', '104', '110', '113', '120', '327', '408']
    uid_pacpvc_val = ['045', '054', '112']
    uid_pacpvc_test = ['002', '038', '039', '052', '053', '069', '070', '075', '078', '090', '100', '419']
    
    # Total UID counts
    total_uid_nsr = uid_nsr_train + uid_nsr_val + uid_nsr_test
    total_uid_af = uid_af_train + uid_af_val + uid_af_test
    total_uid_pacpvc = uid_pacpvc_train + uid_pacpvc_val + uid_pacpvc_test
    total_uid = total_uid_pacpvc + total_uid_af + total_uid_nsr
    
    print('Number of total and unique UIDs:', len(total_uid),'|', len(np.unique(total_uid)))
    print('Number of total and unique NSR UIDs:', len(total_uid_nsr),'|', len(np.unique(total_uid_nsr)))
    print('Number of total and unique AF UIDs:', len(total_uid_af),'|', len(np.unique(total_uid_af)))
    print('Number of total and unique PAC/PVC UIDs:', len(total_uid_pacpvc),'|', len(np.unique(total_uid_pacpvc)))
    
    train_set = uid_nsr_train + uid_af_train + uid_pacpvc_train
    val_set = uid_nsr_val + uid_af_val + uid_pacpvc_val
    test_set = uid_nsr_test + uid_af_test + uid_pacpvc_test
    
    return train_set, val_set, test_set


def split_uids_60_10_30_noPACPVC(pathmaster):
    # ====== Load the per subject arrythmia summary ======
    file_path = pathmaster.summary_path()
    # df_summary = pd.read_csv(file_path)

    # Read the CSV file using pyarrow.csv.read_csv
    table_summary = csv.read_csv(file_path)
    df_summary = table_summary.to_pandas()
    
    df_summary['UID'] = df_summary['UID'].astype(str).str.zfill(3) # Pads each UIDs with enough zeroes to be 3 characters
    
    df_summary['sample_nonAF'] = df_summary['NSR'] + df_summary['PACPVC'] + df_summary['SVT']
    df_summary['sample_AF'] = df_summary['AF']
    
    df_summary['sample_nonAF_ratio'] = df_summary['sample_nonAF'] / (df_summary['sample_AF'] + df_summary['sample_nonAF'])
    
    # Filter out 0-segment UIDs and UIDs without NSR, AF, and/or PAC/PVC
    remaining_UIDs = []
    
    for index, row in df_summary.iterrows():
        UID = row['UID']
        if row['TOTAL'] == 0:
            # There is no segment in this subject, skip this UID.
            print(f'---------UID {UID} has no segments.------------')
        elif (row['NSR'] > 0 or row['AF'] > 0 or row['PACPVC'] > 0): # Append UID only if it contains NSR, AF, or PAC/PVC
            remaining_UIDs.append(UID)
        else:
            print(f'---------UID {UID} has no AF, NSR, or PAC/PVC segments.------------')
        
    # Split UIDs    
    uid_nsr_train = ['003', '020', '024', '041', '044', '047', '049', '050', '058', '063', '077', '084', '088', '091', '098', '099', '106', '109', '111', '118', '325']
    uid_nsr_val = ['014', '030', '036', '074']
    uid_nsr_test = ['011', '012', '027', '035', '037', '055', '056', '057', '083', '094', '101', '119']
    
    uid_af_train = ['017', '302', '306', '307', '310', '311', '319', '321', '324', '400', '402', '405', '406', '407', '409', '410', '415', '420', '421']
    uid_af_val = ['416', '422', '423']
    uid_af_test = ['301', '305', '312', '318', '320', '322', '329', '413', '414']
    
    uid_pacpvc_train = [] # ['005', '007', '013', '021', '022', '026', '028', '029', '042', '064', '068', '073', '080', '086', '087', '089', '093', '104', '110', '113', '120', '327', '408']
    uid_pacpvc_val = [] # ['045', '054', '112']
    uid_pacpvc_test = [] # ['002', '038', '039', '052', '053', '069', '070', '075', '078', '090', '100', '419']
    
    # Total UID counts
    total_uid_nsr = uid_nsr_train + uid_nsr_val + uid_nsr_test
    total_uid_af = uid_af_train + uid_af_val + uid_af_test
    total_uid_pacpvc = uid_pacpvc_train + uid_pacpvc_val + uid_pacpvc_test
    total_uid = total_uid_pacpvc + total_uid_af + total_uid_nsr
    
    print('Number of total and unique UIDs:', len(total_uid),'|', len(np.unique(total_uid)))
    print('Number of total and unique NSR UIDs:', len(total_uid_nsr),'|', len(np.unique(total_uid_nsr)))
    print('Number of total and unique AF UIDs:', len(total_uid_af),'|', len(np.unique(total_uid_af)))
    print('Number of total and unique PAC/PVC UIDs:', len(total_uid_pacpvc),'|', len(np.unique(total_uid_pacpvc)))
    
    train_set = uid_nsr_train + uid_af_train + uid_pacpvc_train
    val_set = uid_nsr_val + uid_af_val + uid_pacpvc_val
    test_set = uid_nsr_test + uid_af_test + uid_pacpvc_test
    
    return train_set, val_set, test_set


def split_uids_60_10_30_noNSR(pathmaster):
    # ====== Load the per subject arrythmia summary ======
    file_path = pathmaster.summary_path()
    # df_summary = pd.read_csv(file_path)

    # Read the CSV file using pyarrow.csv.read_csv
    table_summary = csv.read_csv(file_path)
    df_summary = table_summary.to_pandas()
    
    df_summary['UID'] = df_summary['UID'].astype(str).str.zfill(3) # Pads each UIDs with enough zeroes to be 3 characters
    
    df_summary['sample_nonAF'] = df_summary['NSR'] + df_summary['PACPVC'] + df_summary['SVT']
    df_summary['sample_AF'] = df_summary['AF']
    
    df_summary['sample_nonAF_ratio'] = df_summary['sample_nonAF'] / (df_summary['sample_AF'] + df_summary['sample_nonAF'])
    
    # Filter out 0-segment UIDs and UIDs without NSR, AF, and/or PAC/PVC
    remaining_UIDs = []
    
    for index, row in df_summary.iterrows():
        UID = row['UID']
        if row['TOTAL'] == 0:
            # There is no segment in this subject, skip this UID.
            print(f'---------UID {UID} has no segments.------------')
        elif (row['NSR'] > 0 or row['AF'] > 0 or row['PACPVC'] > 0): # Append UID only if it contains NSR, AF, or PAC/PVC
            remaining_UIDs.append(UID)
        else:
            print(f'---------UID {UID} has no AF, NSR, or PAC/PVC segments.------------')
        
    # Split UIDs    
    uid_nsr_train = [] # ['003', '020', '024', '041', '044', '047', '049', '050', '058', '063', '077', '084', '088', '091', '098', '099', '106', '109', '111', '118', '325']
    uid_nsr_val = [] # ['014', '030', '036', '074']
    uid_nsr_test = [] # ['011', '012', '027', '035', '037', '055', '056', '057', '083', '094', '101', '119']
    
    uid_af_train = ['017', '302', '306', '307', '310', '311', '319', '321', '324', '400', '402', '405', '406', '407', '409', '410', '415', '420', '421']
    uid_af_val = ['416', '422', '423']
    uid_af_test = ['301', '305', '312', '318', '320', '322', '329', '413', '414']
    
    uid_pacpvc_train = ['005', '007', '013', '021', '022', '026', '028', '029', '042', '064', '068', '073', '080', '086', '087', '089', '093', '104', '110', '113', '120', '327', '408']
    uid_pacpvc_val = ['045', '054', '112']
    uid_pacpvc_test = ['002', '038', '039', '052', '053', '069', '070', '075', '078', '090', '100', '419']
    
    # Total UID counts
    total_uid_nsr = uid_nsr_train + uid_nsr_val + uid_nsr_test
    total_uid_af = uid_af_train + uid_af_val + uid_af_test
    total_uid_pacpvc = uid_pacpvc_train + uid_pacpvc_val + uid_pacpvc_test
    total_uid = total_uid_pacpvc + total_uid_af + total_uid_nsr
    
    print('Number of total and unique UIDs:', len(total_uid),'|', len(np.unique(total_uid)))
    print('Number of total and unique NSR UIDs:', len(total_uid_nsr),'|', len(np.unique(total_uid_nsr)))
    print('Number of total and unique AF UIDs:', len(total_uid_af),'|', len(np.unique(total_uid_af)))
    print('Number of total and unique PAC/PVC UIDs:', len(total_uid_pacpvc),'|', len(np.unique(total_uid_pacpvc)))
    
    train_set = uid_nsr_train + uid_af_train + uid_pacpvc_train
    val_set = uid_nsr_val + uid_af_val + uid_pacpvc_val
    test_set = uid_nsr_test + uid_af_test + uid_pacpvc_test
    
    return train_set, val_set, test_set


def split_uids_60_10_30_balanced(pathmaster):
    # ====== Load the per subject arrythmia summary ======
    file_path = pathmaster.summary_path()
    # df_summary = pd.read_csv(file_path)

    # Read the CSV file using pyarrow.csv.read_csv
    table_summary = csv.read_csv(file_path)
    df_summary = table_summary.to_pandas()
    
    df_summary['UID'] = df_summary['UID'].astype(str).str.zfill(3) # Pads each UIDs with enough zeroes to be 3 characters
    
    df_summary['sample_nonAF'] = df_summary['NSR'] + df_summary['PACPVC'] + df_summary['SVT']
    df_summary['sample_AF'] = df_summary['AF']
    
    df_summary['sample_nonAF_ratio'] = df_summary['sample_nonAF'] / (df_summary['sample_AF'] + df_summary['sample_nonAF'])
    
    # Filter out 0-segment UIDs and UIDs without NSR, AF, and/or PAC/PVC
    remaining_UIDs = []
    
    for index, row in df_summary.iterrows():
        UID = row['UID']
        if row['TOTAL'] == 0:
            # There is no segment in this subject, skip this UID.
            print(f'---------UID {UID} has no segments.------------')
        elif (row['NSR'] > 0 or row['AF'] > 0 or row['PACPVC'] > 0): # Append UID only if it contains NSR, AF, or PAC/PVC
            remaining_UIDs.append(UID)
        else:
            print(f'---------UID {UID} has no AF, NSR, or PAC/PVC segments.------------')
        
    # Split UIDs    
    uid_nsr_train = ['041', '044', '047', '050', '058', '063', '091', '098', '106', '111', '325']
    uid_nsr_val = ['014', '030', '036', '074']
    uid_nsr_test = ['011', '012', '027', '035', '037', '055', '056', '057', '083', '094', '101', '119']
    
    uid_af_train = ['017', '302', '306', '307', '310', '311', '319', '321', '324', '400', '402', '407', '409', '415', '420', '421']
    uid_af_val = ['416', '422', '423']
    uid_af_test = ['301', '305', '312', '318', '320', '322', '329', '413', '414']
    
    uid_pacpvc_train = ['005', '007', '013', '021', '022', '026', '028', '029', '042', '064', '068', '073', '080', '086', '087', '089', '093', '104', '110', '113', '120', '327', '408']
    uid_pacpvc_val = ['045', '054', '112']
    uid_pacpvc_test = ['002', '038', '039', '052', '053', '069', '070', '075', '078', '090', '100', '419']
    
    # Total UID counts
    total_uid_nsr = uid_nsr_train + uid_nsr_val + uid_nsr_test
    total_uid_af = uid_af_train + uid_af_val + uid_af_test
    total_uid_pacpvc = uid_pacpvc_train + uid_pacpvc_val + uid_pacpvc_test
    total_uid = total_uid_pacpvc + total_uid_af + total_uid_nsr
    
    print('Number of total and unique UIDs:', len(total_uid),'|', len(np.unique(total_uid)))
    print('Number of total and unique NSR UIDs:', len(total_uid_nsr),'|', len(np.unique(total_uid_nsr)))
    print('Number of total and unique AF UIDs:', len(total_uid_af),'|', len(np.unique(total_uid_af)))
    print('Number of total and unique PAC/PVC UIDs:', len(total_uid_pacpvc),'|', len(np.unique(total_uid_pacpvc)))
    
    train_set = uid_nsr_train + uid_af_train + uid_pacpvc_train
    val_set = uid_nsr_val + uid_af_val + uid_pacpvc_val
    test_set = uid_nsr_test + uid_af_test + uid_pacpvc_test
    
    return train_set, val_set, test_set


def split_uids(pathmaster):
    # ====== Load the per subject arrythmia summary ======
    file_path = pathmaster.summary_path()
    # df_summary = pd.read_csv(file_path)

    # Read the CSV file using pyarrow.csv.read_csv
    table_summary = csv.read_csv(file_path)
    df_summary = table_summary.to_pandas()
    
    df_summary['UID'] = df_summary['UID'].astype(str).str.zfill(3) # Pads each UIDs with enough zeroes to be 3 characters
    
    df_summary['sample_nonAF'] = df_summary['NSR'] + df_summary['PACPVC'] + df_summary['SVT']
    df_summary['sample_AF'] = df_summary['AF']
    
    df_summary['sample_nonAF_ratio'] = df_summary['sample_nonAF'] / (df_summary['sample_AF'] + df_summary['sample_nonAF'])
    
    all_UIDs = df_summary['UID'].unique()
   
    # ====================================================
    # ====== AF trial separation ======
    # R:\ENGR_Chon\Dong\Numbers\Pulsewatch_numbers\Fahimeh_CNNED_general_ExpertSystemwApplication\tbl_file_name\TrainingSet_final_segments
    AF_trial_Fahimeh_train = ['402','410']
    AF_trial_Fahimeh_test = ['301', '302', '305', '306', '307', '310', '311', 
                             '312', '318', '319', '320', '321', '322', '324', 
                             '325', '327', '329', '400', '406', '407', '409',
                             '414']
    AF_trial_Fahimeh_did_not_use = ['405', '413', '415', '416', '420', '421', '422', '423']
    AF_trial_paroxysmal_AF = ['408','419']
    
    AF_trial_train = AF_trial_Fahimeh_train
    AF_trial_test = AF_trial_Fahimeh_test
    AF_trial_unlabeled = AF_trial_Fahimeh_did_not_use + AF_trial_paroxysmal_AF
    print(f'AF trial: {len(AF_trial_train)} training subjects {AF_trial_train}')
    print(f'AF trial: {len(AF_trial_test)} testing subjects {AF_trial_test}')
    print(f'AF trial: {len(AF_trial_unlabeled)} unlabeled subjects {AF_trial_unlabeled}')
  
    # =================================
    # === Clinical trial AF subjects separation ===
    clinical_trial_AF_subjects = ['005', '017', '026', '051', '075', '082']
    
    # Filter out AF trial and 0-segment UIDs
    remaining_UIDs = []
    count_NSR = []
    
    for index, row in df_summary.iterrows():
        UID = row['UID']
        this_NSR = row['sample_nonAF']
        if math.isnan(row['sample_nonAF_ratio']): # sample_nonAF is never NaN, sample_nonAF_ratio may be NaN
            # There is no segment in this subject, skip this UID.
            print(f'---------UID {UID} has no segments.------------')
            continue # If a UID has no segments, skip the rest of the for loop for this index, row
        if UID not in AF_trial_train and UID not in AF_trial_test and UID not in clinical_trial_AF_subjects \
            and UID[0] != '3' and UID[0] != '4':
            remaining_UIDs.append(UID)
            count_NSR.append(this_NSR)
    
    # From the candidate UIDs, select a subset to be used for training, validation, and testing
    random.seed(seed=42)
    
    list_of_candidates = remaining_UIDs
    number_of_items_to_pick = round(len(list_of_candidates) * 0.25) # 15% labeled for training, 10% for testing.
    sum_NSR = sum(count_NSR)
    
    # probability_distribution = [x/sum_NSR for x in count_NSR] # Proportion of total NSR segments for each UID
    probability_distribution = [(1-x/sum_NSR)/ (len(count_NSR)-1) for x in count_NSR] # Subjects with fewer segments have higher chance to be selected.
    draw = choice(list_of_candidates, number_of_items_to_pick,
                  p=probability_distribution, replace=False)
    
    # Ensures that training set contains both AF and non-AF
    clinical_trial_train_nonAF = list(draw[:round(len(list_of_candidates) * 0.12)]) # Draws the first X number of candidates equal to 7% of the total list of candidates
    clinical_trial_train_temp = clinical_trial_train_nonAF + clinical_trial_AF_subjects[:round(len(clinical_trial_AF_subjects)/2)]
    clinical_trial_train = []
    
    for UID in clinical_trial_train_temp:
        # UID 051 and 108 and maybe other UIDs had no segments (unknown reason).
        if UID in all_UIDs:
            clinical_trial_train.append(UID) # Only use the UIDs that are in the summary to test
    
    # Ensures that the testing set contains both AF and non-AF
    clinical_trial_test_nonAF = list(draw[round(len(list_of_candidates) * 0.12):]) # Draws the remaining candidates
    clinical_trial_test_temp = clinical_trial_test_nonAF + clinical_trial_AF_subjects[round(len(clinical_trial_AF_subjects)/2):]
    clinical_trial_test = []
    for UID in clinical_trial_test_temp:
        # UID 051 and 108 and maybe other UIDs had no segments (unknown reason).
        if UID in all_UIDs:
            clinical_trial_test.append(UID) # Only use the UIDs that are in the summary to test
    
    # Uses all remaining subset of UIDs from original list not used in training or validating for testing
    clinical_trial_unlabeled = []
    for UID in remaining_UIDs: # Changed from all_UIDs to remove UIDs with 0 segments (i.e. UID 108)
        if UID not in clinical_trial_train and UID not in clinical_trial_test and UID[0] != '3' and UID[0] != '4':
            clinical_trial_unlabeled.append(UID)
    
    # Sum up to 74 UIDs, all of the ones that do not start with '3' or '4' and dropping UID 108 which has 0 segments
    print(f'Clinical trial: selected {len(clinical_trial_train)} UIDs for training {clinical_trial_train}') # Contains both non-AF and AF clinical trial subjects
    print(f'Clinical trial: selected {len(clinical_trial_test)} UIDs for testing {clinical_trial_test}') # Contains both non-AF and AF clinical trial subjects
    print(f'Clinical trial: selected {len(clinical_trial_unlabeled)} UIDs for unlabeled {clinical_trial_unlabeled}') # All remaining clinical trial subjects...probably contains both AF and non-AF
    
    # Used to make sure the model runs correctly
    clinical_trial_train = ['063','416','005'] # Training
    clinical_trial_test = ['058','409','054'] # Evaluation
    clinical_trial_unlabeled = ['029','036','421'] # Testing
    
    return clinical_trial_train, clinical_trial_test, clinical_trial_unlabeled


def extract_uids_from_directory(directory_path):
    uids = set()  # Use a set to avoid duplicate UIDs
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory_path):
        uid = filename[:3]  # Extract the first 3 characters (UID)
        uids.add(str(uid))  # Add UID to the set
    
    return list(uids)  # Convert the set to a list


def mimic3_uids(pathmaster):
    _, labels_path = pathmaster.mimic3_paths()
    UIDs = extract_uids_from_directory(labels_path)
    
    return UIDs   
    
    
def simband_uids(pathmaster):
    _, labels_path = pathmaster.simband_paths()
    UIDs = extract_uids_from_directory(labels_path)      
    
    return UIDs
    

class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, UIDs, standardize=True, data_format='csv', read_all_labels=False, 
                 start_idx=0, img_channels=1, img_size=128, downsample=None, data_type=torch.float32, is_tfs=True, binary=False):
        self.data_path = data_path
        self.labels_path = labels_path
        self.UIDs = UIDs
        self.standardize = standardize
        self.data_format = data_format
        self.read_all_labels = read_all_labels
        self.transforms = ToTensor()
        self.start_idx = start_idx  # Initial batch index to start from, useful for resuming training
        self.img_channels = img_channels
        self.img_size = img_size
        self.downsample = downsample
        self.is_tfs = is_tfs
        self.binary = binary
        
        # Must be manually set so that the image resolution chosen is the one that is returned
        self.dtype = data_type
        
        self.refresh_dataset()

    def refresh_dataset(self):
        self.segment_names, self.labels = self.extract_segment_names_and_labels()

    def add_uids(self, new_uids):
        unique_new_uids = [uid for uid in new_uids if uid not in self.UIDs] # Appends any unqiue new UID in self.UIDs to unique_new_uids
        self.UIDs.extend(unique_new_uids) # Appends unique_new_uids to UIDs
        self.refresh_dataset()

    def __len__(self): # Method is implicitly called when len() is used on an instance of CustomDataset
        return len(self.segment_names)

    def save_checkpoint(self, checkpoint_path): # Likely not worth using, simply use the save_checkpoint() function in train_func.py 
        # Enhanced to automatically include 'start_idx' in the checkpoint
        checkpoint = {
            'segment_names': self.segment_names,
            'labels': self.labels,
            'UIDs': self.UIDs,
            'start_idx': self.start_idx  # Now also saving start_idx
        }
        torch.save(checkpoint, checkpoint_path) #  Using standard Python methods like pickle or json is generally recommended for dictionaries, there are no benefits for using torch.save, no real harm either

    def load_checkpoint(self, checkpoint_path): # Reloads where you started off last time (not where you ended), just use analogous function in train_func.py
        checkpoint = torch.load(checkpoint_path)
        self.segment_names = checkpoint['segment_names'] # Seems redundant since it is overwritten by refresh_dataset()
        self.labels = checkpoint['labels'] # Seems redundant since it is overwritten by refresh_dataset()
        self.UIDs = checkpoint['UIDs']
        # Now also loading and setting start_idx from checkpoint
        self.start_idx = checkpoint.get('start_idx', 0) # Returns 0 if no start_idx found
        self.refresh_dataset()

    def __getitem__(self, idx): # Method is implicitly called when getitem() is used on an instance of CustomDataset. It is called batch_size number of times per iteration of dataloader | Loads segments as needed (lazy loading)
        actual_idx = (idx + self.start_idx) % len(self.segment_names)  # Adjust index based on start_idx and wrap around if needed (i.e. index falls out of bounds)
        segment_name = self.segment_names[actual_idx]
        label = self.labels[segment_name]

        if hasattr(self, 'all_data') and actual_idx < len(self.all_data): # When Luis uses adds data to train_loader in main_checkpoints.py, 
        # new data is added (creating all_data) only after train_loader is created with its original training data. This means that if self.all_data
        # exists, then __getitem__ is only be called in order to retrieve data newly added to train_loader in all_data
            time_freq_tensor = self.all_data[actual_idx]
        else:
            time_freq_tensor = self.load_data(segment_name)

        return {'data': time_freq_tensor, 'label': label, 'segment_name': segment_name}
    
        # When iterating over the dataloader, which returns batches of data, each batch will contain a dictionary with keys corresponding to the data and labels.

        # Since the dataloader's dataset's __getitem__ method returns a dictionary with keys 'data', 'label', and 'segment_name', the returned batch will be a dictionary where:

        # The 'data' key will correspond to a tensor of shape (batch_size, ...), representing the shape of the data.
        # The 'label' key will correspond to a tensor of shape (batch_size, ...), representing the shape of the labels.
        # The 'segment_name' key will correspond to a tensor of shape (batch_size, ...), representing the shape of the segment_name.

    def set_start_idx(self, index):
        self.start_idx = index
      
    def add_data_label_pair(self, data, label):
        # Assign a unique ID or name for the new data
        new_id = len(self.segment_names)
        segment_name = f"new_data_{new_id}"

        # Append the new data and label
        self.segment_names.append(segment_name)
        self.labels[segment_name] = label

        # Append the new data tensor to an attribute that holds all of the newly added data
        if hasattr(self, 'all_data'):
            self.all_data.append(data)
        else:
            self.all_data = [data]
    
    
    def extract_segment_names_and_labels(self): # Only extract the segments and labels of a particular class.
        segment_names = []
        labels = {}
        
        # If a subject is not loading and there are no errors, just these lists
        uid_nsr = ['011', '014', '041', '050', '056', '058', '083', '106', '109',
                   '037', '047', '055', '074', '091', '098', '101', '119', '325',
                   '003', '012', '020', '024', '027', '030', '035', '036', '044', '049', '057', '063', '077', '084', '088', '094', '099', '111', '118']
        uid_af = ['305', '307', '311', '318', '320', '322', '405', '415', '423',
                  '301', '319', '321', '324', '329', '400', '406', '409', '416',
                  '017', '302', '306', '310', '312', '402', '407', '410', '413', '414', '420', '421', '422']
        uid_pacpvc = ['007', '022', '028', '038', '054', '068', '075', '086', '087', '093', '120', '327',
                      '002', '005', '013', '021', '026', '029', '045', '073', '089', '100', '112', '408',
                      '039', '042', '052', '053', '064', '069', '070', '078', '080', '090', '104', '110', '113', '419']
        for UID in self.UIDs:
            if UID[0] == '6':
                label_file = os.path.join(self.labels_path, UID + "_GT_20240425.csv")
            elif UID[0] == '7':
                label_file = os.path.join(self.labels_path, UID + "_GT_20240809.csv")
            else:
                label_file = os.path.join(self.labels_path, UID + "_final_attemp_4_1_Dong.csv")
            if os.path.exists(label_file):
                # label_data = pd.read_csv(label_file, sep=',', header=0, names=['segment', 'label']) # Replaces the original headers with names
                
                # Use PyArrow to read csv
                parse_options = csv.ParseOptions(delimiter=',') # Indicate delimiter
                read_options = csv.ReadOptions(column_names=['segment', 'label'], skip_rows=1) # Assign desired column names and skip the first row (headers)
                label_data = csv.read_csv(label_file, parse_options=parse_options, read_options=read_options)
                label_data = label_data.to_pandas()
                
                label_segment_names = label_data['segment'].apply(lambda x: x.split('.')[0]) # Splits each segment name by '.' and retrieves the first part
                if len(label_segment_names) == 0: # Checks if list is empty
                    label_segment_names = label_data['segment']
                
                for idx, segment_name in enumerate(label_segment_names): # enumerate() returns the value and corresponding index of each element in an iterable
                    label_val = label_data['label'].values[idx]
                    # Will only use NSR (0), AF (1), and PAC/PVC(2) and not SVT (3)
                    if self.read_all_labels: # If reading all labels, set all labels not 0, 1, or 2 to -1 and return all labels
                        # Assign -1 if label is not in [0, 1, 2]
                        labels[segment_name] = label_val if label_val in [0, 1, 2] else -1
                        if segment_name not in segment_names:
                            segment_names.append(segment_name)
                    else:
                        # Only add segments with labels in [0, 1, 2]
                        if label_val in [0, 1, 2] and segment_name not in segment_names:
                            # Temporary solution to ensure only segments of a particular class are loaded for each UID
                            if UID[0] == '6' or UID[0] == '7':
                                segment_names.append(segment_name)
                                labels[segment_name] = label_val
                            else:
                                if UID in uid_nsr and label_val == 0:
                                    segment_names.append(segment_name)
                                    labels[segment_name] = label_val
                                elif UID in uid_af and label_val == 1:
                                    segment_names.append(segment_name)
                                    labels[segment_name] = label_val
                                elif UID in uid_pacpvc and label_val == 2:
                                    segment_names.append(segment_name)
                                    if self.binary:
                                        labels[segment_name] = 0
                                    else:
                                        labels[segment_name] = label_val
                          
        return segment_names, labels
    

    def load_data(self, segment_name):
        data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0])
        if self.is_tfs:
            seg_path = os.path.join(data_path_UID, segment_name + '_filt_STFT.' + self.data_format)
        else:
            seg_path = os.path.join(data_path_UID, segment_name + '_density_poincare.' + self.data_format)
            

        try: # Allows to define a block of code to be executed and specify how to handle any errors that might occur during its execution
            if self.data_format == 'csv' and seg_path.endswith('.csv'):
                # time_freq_plot = np.array(pd.read_csv(seg_path, header=None))
                
                # Use PyArrow to read csv
                read_options = csv.ReadOptions(autogenerate_column_names=True)
                seg_data = csv.read_csv(seg_path, read_options=read_options)
                time_freq_plot = seg_data.to_pandas().to_numpy()
                
                time_freq_tensor = torch.tensor(time_freq_plot).reshape(self.img_channels, self.img_size, self.img_size)
            elif self.data_format == 'png' and seg_path.endswith('.png'):
                img = Image.open(seg_path)
                img_data = np.array(img)
                time_freq_tensor = torch.tensor(img_data).unsqueeze(0)
            elif self.data_format == 'pt' and seg_path.endswith('.pt'):
                time_freq_tensor = torch.load(seg_path)
            else:
                raise ValueError("Unsupported file format")

            if self.downsample is not None:
                # Downsample the image
                # Use OpenCV to resize the array to downsample x downsample using INTER_AREA interpolation
                time_freq_array = cv2.resize(np.array(time_freq_tensor.reshape(self.img_size, self.img_size).to('cpu')), (self.downsample, self.downsample), interpolation=cv2.INTER_AREA)
                time_freq_tensor = torch.tensor(time_freq_array, dtype=self.dtype).reshape(self.img_channels, self.downsample, self.downsample)
            else:
                time_freq_tensor = time_freq_tensor.reshape(self.img_channels, self.img_size, self.img_size).to(self.dtype)

            if self.standardize:
                time_freq_tensor = self.standard_scaling(time_freq_tensor) # Standardize the data
            
            return time_freq_tensor

        except Exception as e:
            print(f"Error processing segment: {segment_name}. Exception: {str(e)}")
            traceback.print_exc()
            if self.downsample is not None:
                return torch.zeros((self.img_channels, self.downsample, self.downsample))  # Return zeros in case of an error
            else:
                return torch.zeros((self.img_channels, self.img_size, self.img_size))  # Return zeros in case of an error

    def standard_scaling(self, data):
        data = data.cpu()
        scaler = StandardScaler()
        data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape) # Converts data into 2D array, standardizes it, reshapes it back into 3D (1,X,X)
        return torch.tensor(data, dtype=self.dtype)

def load_data_split_batched(data_path, labels_path, UIDs, batch_size, standardize=False, data_format='csv', 
                            read_all_labels=False, drop_last=False, num_workers=4, start_idx=0, 
                            img_channels=1, img_size=128, downsample=None, data_type=torch.float32, is_tfs=True, binary=False):
    torch.manual_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
    
    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True
    
    dataset = CustomDataset(data_path, labels_path, UIDs, standardize, data_format, read_all_labels, start_idx=start_idx, 
                            img_channels=img_channels, img_size=img_size, downsample=downsample, data_type=data_type, is_tfs=is_tfs, binary=binary)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g) # Prefetches 2 batches ahead of current training iteration (allows loading of data simultaneously with training). Shuffle is set to False to resume training at a specific batch.
    return dataloader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Function to extract and preprocess data
def preprocess_data(data_format, clinical_trial_train, clinical_trial_test, clinical_trial_unlabeled, batch_size, standardize=False,
                    read_all_labels=False, img_channels=1, img_size=128, downsample=None, data_type=torch.float32, pathmaster=None, binary=False):
    start_idx = 0
    # data_path, labels_path = pathmaster.data_paths(data_format)
    data_path = pathmaster.data_path
    labels_path = pathmaster.labels_path
    
    if data_format == 'csv':
        num_workers = 6
    elif data_format == 'pt':
        num_workers = 8
    
    train_loader = load_data_split_batched(data_path, labels_path, clinical_trial_train, batch_size, standardize=standardize, 
                                           data_format=data_format, read_all_labels=read_all_labels, num_workers=num_workers,
                                           start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                           data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary)
    val_loader = load_data_split_batched(data_path, labels_path, clinical_trial_test, batch_size, standardize=standardize, 
                                         data_format=data_format, read_all_labels=read_all_labels, num_workers=num_workers, 
                                         start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                         data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary)
    test_loader = load_data_split_batched(data_path, labels_path, clinical_trial_unlabeled, batch_size, standardize=standardize, 
                                          data_format=data_format, read_all_labels=read_all_labels, num_workers=num_workers,
                                          start_idx=start_idx, img_channels=img_channels, img_size=img_size, downsample=downsample,
                                          data_type=data_type, is_tfs=pathmaster.is_tfs, binary=binary)
    return train_loader, val_loader, test_loader

def map_samples_to_uids(uncertain_sample_indices, dataset):
    """
    Maps indices of uncertain samples back to their corresponding segment names or UIDs.

    Args:
    - uncertain_sample_indices: Indices of the uncertain samples in the dataset.
    - dataset: The dataset object which contains the mapping of segment names and UIDs.

    Returns:
    - List of UIDs or segment names corresponding to the uncertain samples.
    """
    return [dataset.segment_names[i] for i in uncertain_sample_indices]
    