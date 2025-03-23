# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:06:14 2024

@author: dchen
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from imblearn.metrics import classification_report_imbalanced

# For increased csv speed
import pyarrow as pa
from pyarrow import csv

def save_hyperparameters(hyperparameters, pathmaster):
    hyperparameters_path = pathmaster.hyperparameters_path()
    hyperparameters_path = os.path.join(hyperparameters_path, 'hyperparameters_' + pathmaster.file_tag + '.csv')

    # If there are class weights, make sure all other columns have same length
    if hyperparameters['class_weights'] is not None:
        # Update the dictionary
        for key, value in hyperparameters.items():
        # If the length of the value is less than max_length
            if key != 'class_weights':
                # Fill missing values with np.nan
                hyperparameters[key] = [value] + [np.nan] * (len(hyperparameters['class_weights']) - 1)
    
    hyperparameters = pd.DataFrame(hyperparameters) 
    hyperparameters.to_csv(hyperparameters_path, index=False)
    
    # # Using PyArrow (need each hyperparameter to be a list)
    # hyperparameters_table = pa.Table.from_pydict(hyperparameters)
    # csv.write_csv(hyperparameters_table, hyperparameters_path)


def save_losslists(losslist_train, losslist_val, pathmaster): # For holdout training and validation
    losslists_path = pathmaster.losslists_path()
    losslists_path = os.path.join(losslists_path, 'losslists_' + pathmaster.file_tag + '.csv')
    # losslists = pd.DataFrame(dtype='float32')
    # losslists['training'] = losslist_train
    # losslists['validation'] = losslist_val
    # losslists.to_csv(losslists_path, index=False, chunksize=500)
    
    # Using PyArrow
    # losslists = {
    #     'training': losslist_train, 
    #     'validation': losslist_val
    #     }
    # losslists_table = pa.Table.from_pydict(losslists)
    losslists = [np.array(losslist_train).reshape(-1).astype(np.float32), np.array(losslist_val).reshape(-1).astype(np.float32)]
    losslists_names = ['training', 'validation']
    losslists_table = pa.Table.from_arrays(losslists, losslists_names)
    csv.write_csv(losslists_table, losslists_path)
    
    
def save_runtime_list(epoch_time_list, pathmaster):
    # epoch_time_array = np.array(epoch_time_list).reshape(-1).astype(np.float32)
    runtime_lists_path = pathmaster.runtime_lists_path()
    runtime_lists_path = os.path.join(runtime_lists_path, 'runtime_lists_' + pathmaster.file_tag + '.csv')
    # runtime_list = pd.DataFrame(dtype='float32')
    # runtime_list['time_sec'] = epoch_time_list
    # runtime_list.to_csv(runtime_lists_path, index=False, chunksize=500)
    
    # Using PyArrow
    runtime_dict = {'epoch_time_sec': epoch_time_list,
                    'mean_time_sec': [sum(epoch_time_list)/len(epoch_time_list)] + [np.nan] * (len(epoch_time_list) - 1)}
    runtime_table = pa.Table.from_pydict(runtime_dict)
    # runtime_table = pa.Table.from_arrays([epoch_time_array, np.array([np.mean(epoch_time_array)])], names=['epoch_time_sec', 'mean_time_sec'])
    csv.write_csv(runtime_table, runtime_lists_path)
    

def save_labels(labels, pathmaster):
    labels = labels.astype(np.int8)
    ground_truths_path = pathmaster.ground_truths_path()
    ground_truths_path = os.path.join(ground_truths_path, 'labels_' + pathmaster.file_tag + '.csv')
    # labels = pd.DataFrame(np.array(labels), dtype='int')
    # labels.to_csv(ground_truths_path, index=False, chunksize=500)
    
    # Using PyArrow
    # labels_dict = {'labels': labels.reshape(-1)} # Convert to 1D array
    # labels_table = pa.Table.from_pydict(labels_dict)
    labels_table = pa.Table.from_arrays([labels.reshape(-1)], names=['labels'])
    csv.write_csv(labels_table, ground_truths_path)


def save_predictions(predictions, pathmaster):
    predictions = predictions.astype(np.int8)
    predictions_path = pathmaster.predictions_path()
    predictions_path = os.path.join(predictions_path, 'predictions_' + pathmaster.file_tag + '.csv')
    # predictions = pd.DataFrame(np.array(predictions), dtype='int')
    # predictions.to_csv(predictions_path, index=False, chunksize=500)
    
    # Using PyArrow
    # predictions_dict = {'predictions': predictions.reshape(-1)} # Convert to 1D array
    # predictions_table = pa.Table.from_pydict(predictions_dict)
    predictions_table = pa.Table.from_arrays([predictions.reshape(-1)], names=['predictions'])
    csv.write_csv(predictions_table, predictions_path)
    
    
def save_prediction_proba(prediction_proba, pathmaster):
    prediction_proba = prediction_proba.astype(np.float32)
    prediction_proba_path = pathmaster.prediction_proba_path()
    prediction_proba_path = os.path.join(prediction_proba_path, 'prediction_proba_' + pathmaster.file_tag + '.csv')
    # prediction_proba = pd.DataFrame(np.array(prediction_proba), dtype='float32')
    # prediction_proba.to_csv(prediction_proba_path, index=False, chunksize=500)
    
    # Using PyArrow
    # # Create PyArrow arrays with specific data type (float64)
    # prediction_proba_dict = {
    #     '0': prediction_proba[:,0],
    #     '1': prediction_proba[:,1],
    #     '2': prediction_proba[:,2]
    #     }

    # Create a PyArrow table
    # prediction_proba_Table = pa.Table.from_pydict(prediction_proba_dict)
    # col_arrays = [prediction_proba[:,0], prediction_proba[:,1]]
    # prediction_proba_Table = pa.Table.from_arrays(col_arrays, names=['0', '1'])
    # csv.write_csv(prediction_proba_Table, prediction_proba_path)
    col_arrays = [prediction_proba[:,0], prediction_proba[:,1], prediction_proba[:,2]]
    prediction_proba_Table = pa.Table.from_arrays(col_arrays, names=['0', '1', '2'])
    csv.write_csv(prediction_proba_Table, prediction_proba_path)
    
    
def save_prediction_proba_binary(prediction_proba, pathmaster):
    prediction_proba = prediction_proba.astype(np.float32)
    prediction_proba_path = pathmaster.prediction_proba_path()
    prediction_proba_path = os.path.join(prediction_proba_path, 'prediction_proba_' + pathmaster.file_tag + '.csv')
    # prediction_proba = pd.DataFrame(np.array(prediction_proba), dtype='float32')
    # prediction_proba.to_csv(prediction_proba_path, index=False, chunksize=500)
    
    # Using PyArrow
    # # Create PyArrow arrays with specific data type (float64)
    # prediction_proba_dict = {
    #     '0': prediction_proba[:,0],
    #     '1': prediction_proba[:,1],
    #     '2': prediction_proba[:,2]
    #     }

    # Create a PyArrow table
    # prediction_proba_Table = pa.Table.from_pydict(prediction_proba_dict)
    # col_arrays = [prediction_proba[:,0], prediction_proba[:,1]]
    # prediction_proba_Table = pa.Table.from_arrays(col_arrays, names=['0', '1'])
    # csv.write_csv(prediction_proba_Table, prediction_proba_path)
    col_arrays = [prediction_proba[:,0], prediction_proba[:,1]]
    prediction_proba_Table = pa.Table.from_arrays(col_arrays, names=['0', '1'])
    csv.write_csv(prediction_proba_Table, prediction_proba_path)
        
    
def metrics(Y_true, Y_pred, Y_proba, save=False, pathmaster=None):
    averages = ['micro', 'macro', 'weighted']
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    
    for average in averages:
        accuracy = accuracy_score(Y_true, Y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average=average)
        auc = roc_auc_score(Y_true, Y_proba, average=average, multi_class='ovr')
        
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)
    
    metrics = {
        'accuracy': accuracy_list,
        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list,
        'auc': auc_list
        }
    
    if save:
        metrics_path = pathmaster.metrics_path()
        metrics_path = os.path.join(metrics_path, 'metrics_' + pathmaster.file_tag + '.csv')
        # metrics = pd.DataFrame(metrics, index=[0], dtype='float32') 
        # metrics.to_csv(metrics_path, index=False)
        
        # Using PyArrow
        metrics_table = pa.Table.from_pydict(metrics)
        csv.write_csv(metrics_table, metrics_path)
        
        
def metrics_binary(Y_true, Y_pred, Y_proba, save=False, pathmaster=None):
    averages = ['micro', 'macro', 'weighted']
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    
    for average in averages:
        accuracy = accuracy_score(Y_true, Y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average=average)
        auc = roc_auc_score(Y_true, Y_proba[:,1], average=average)
        
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)
    
    metrics = {
        'accuracy': accuracy_list,
        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list,
        'auc': auc_list
        }
    
    if save:
        metrics_path = pathmaster.metrics_path()
        metrics_path = os.path.join(metrics_path, 'metrics_' + pathmaster.file_tag + '.csv')
        # metrics = pd.DataFrame(metrics, index=[0], dtype='float32') 
        # metrics.to_csv(metrics_path, index=False)
        
        # Using PyArrow
        metrics_table = pa.Table.from_pydict(metrics)
        csv.write_csv(metrics_table, metrics_path)


def save_classification_report(Y_true, Y_pred, save=False, pathmaster=None):
    report = classification_report(Y_true, Y_pred, output_dict=True)
    row_labels = ['precision', 'recall', 'f1', 'support']
    
    if save:
        classification_report_path = pathmaster.classification_report_path()
        classification_report_path = os.path.join(classification_report_path, 'classification_report_' + pathmaster.file_tag + '.csv')
        report = pd.DataFrame(report)
        # report.reset_index(inplace=True)
        report.insert(loc=0, column='metrics', value=row_labels)
        report.to_csv(classification_report_path, index=False)
        
        # # Using PyArrow
        # report_table = pa.Table.from_pydict(report)
        # csv.write_csv(report_table, classification_report_path)
        
        
def save_classification_report_imbalanced(Y_true, Y_pred, save=False, pathmaster=None):
    report_imbalanced = classification_report_imbalanced(Y_true, Y_pred, output_dict=True)
    row_labels = ['precision', 'recall', 'specificity', 'f1', 'geo mean', 'iba', 'support']
    
    if save:
        classification_report_imbalanced_path = pathmaster.classification_report_imbalanced_path()
        classification_report_imbalanced_path = os.path.join(classification_report_imbalanced_path, 'classification_report_imbalanced_' + pathmaster.file_tag + '.csv')
        report_imbalanced = pd.DataFrame(report_imbalanced)
        # report_imbalanced.reset_index(inplace=True)
        report_imbalanced.insert(loc=0, column='metrics', value=row_labels)
        report_imbalanced.to_csv(classification_report_imbalanced_path, index=False)
        
        # # Using PyArrow
        # report_imbalanced_table = pa.Table.from_pydict(report_imbalanced)
        # csv.write_csv(report_imbalanced_table, classification_report_imbalanced_path)      
        

def roc_curves(y_test, y_prob, save=False, pathmaster=None, class_names=['NSR', 'AF', 'PAC/PVC']):
  # Get the unique class labels
  classes = np.unique(y_test)
  
  if class_names is None:
        class_names = np.unique(y_test)

  # Convert labels to binary matrix
  y_bin = label_binarize(y_test, classes=classes)

  # Pre-allocate arrays for ROC curves
  fpr_mean = np.linspace(0, 1, 100)
  tpr_mean = []
  fpr = []
  tpr = []
  AUC = []

  # Calculate ROC curves for each class
  for i, class_label in enumerate(classes):
    fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    AUC.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
    fpr.append(fpr_i)
    tpr.append(tpr_i)

    # Interpolate TPR for mean ROC curve
    tpr_mean.append(np.interp(fpr_mean, fpr_i, tpr_i))

  # Calculate mean and standard deviation for TPR and AUC
  tpr_mean = np.mean(np.array(tpr_mean).reshape(len(classes), -1), axis=0)
  tpr_stdv = np.std(tpr_mean, axis=0)
  mean_auc = auc(fpr_mean, tpr_mean)
  std_auc = np.std(AUC)

  # Create the plot
  plt.figure(figsize=(12, 9))
  plt.clf()
  plt.plot([0, 1], [0, 1], 'k--')
  plt.axis([0, 1, 0, 1])
  plt.xlabel('False Positive Rate', fontsize=16)
  plt.ylabel('True Positive Rate', fontsize=16)
  plt.title('ROC Curves (' + pathmaster.file_tag + ')', fontweight='bold')

  # Plot individual ROC curves
  for i in range(len(classes)):
    label_str = f"ROC Label {class_names[i]} (AUC = {AUC[i]:.3f})"
    plt.plot(fpr[i], tpr[i], linewidth=3, label=label_str)

  # Plot mean ROC curve with standard deviation
  plt.plot(fpr_mean, tpr_mean, color='k', label=rf"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})", linewidth=5)
  plt.fill_between(fpr_mean, np.maximum(tpr_mean - tpr_stdv, 0), np.minimum(tpr_mean + tpr_stdv, 1), color='grey', alpha=0.2, label=r"$\pm$ 1 std. dev.")

  plt.legend(loc="lower right")
  
  if save:
        roc_curves_path = pathmaster.roc_curves_path()
        roc_curves_path = os.path.join(roc_curves_path, 'roc_curves_' + pathmaster.file_tag + '.jpg')
        plt.savefig(roc_curves_path, dpi=150)
    
    
def roc_curves_binary(y_test, y_prob, save=False, pathmaster=None, class_names=['Negative', 'Positive']):
    y_prob = y_prob[:,1]
    # Convert labels to binary matrix
    y_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Pre-allocate arrays for ROC curves
    fpr_mean = np.linspace(0, 1, 100)
    tpr_mean = []
    fpr = []
    tpr = []
    AUC = []

    # Calculate ROC curve for the positive class
    fpr, tpr, _ = roc_curve(y_bin, y_prob)
    AUC = roc_auc_score(y_bin, y_prob)

    # Create the plot
    plt.figure(figsize=(12, 9))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {AUC:.3f})')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")
  
    if save:
        roc_curves_path = pathmaster.roc_curves_path()
        roc_curves_path = os.path.join(roc_curves_path, 'roc_curves_' + pathmaster.file_tag + '.jpg')
        plt.savefig(roc_curves_path, dpi=150)
    

def mean_roc_curves(Y_tests, Y_probas, clf_names, save=False, pathmaster=None):
    # Pre-allocate arrays for ROC curves
    fpr_mean = np.linspace(0, 1, 100)
    # tpr_mean = np.zeros_like(fpr_mean)

    # Set figure size
    plt.figure(figsize=(12,9))
    
    # Plot individual mean ROC curves for each classifier
    for y_test, y_prob, clf_name in zip(Y_tests, Y_probas, clf_names):
        # Get the unique class labels
        classes = np.unique(y_test)
        
        # Convert labels to binary matrix
        y_bin = label_binarize(y_test, classes=classes)

        # Pre-allocate arrays for ROC curves
        fpr = []
        tpr = []
        AUC = []

        # Calculate ROC curves for each class
        for i, class_label in enumerate(classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            AUC.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
            fpr.append(fpr_i)
            tpr.append(tpr_i)

        # Interpolate TPR for mean ROC curve
        tpr_interp = [np.interp(fpr_mean, fpr_i, tpr_i) for fpr_i, tpr_i in zip(fpr, tpr)]
        tpr_mean = np.mean(tpr_interp, axis=0)

        # Plot mean ROC curve
        plt.plot(fpr_mean, tpr_mean, label=f"{clf_name} - Mean ROC (AUC = {auc(fpr_mean, tpr_mean):.3f} $\pm$ {np.std(AUC):.3f})", linewidth=2)

    # Additional plot configurations
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Mean ROC Curve(s)', fontweight='bold')
    plt.legend(loc="lower right")
    # plt.show()
    
    if save:
        mean_roc_curves_path = pathmaster.mean_roc_curves_path()
        mean_roc_curves_path = os.path.join(mean_roc_curves_path, 'mean_roc_curves_' + pathmaster.file_tag + '.jpg')
        plt.savefig(mean_roc_curves_path, dpi=150)
    

def conf_matrix(conf_matrix, title='Confusion Matrix', save=False, pathmaster=None, class_names=['NSR', 'AF', 'PAC/PVC'], text_size_factor=2): 
    title = title + ' (' + pathmaster.file_tag + ')'
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Normalize

    plt.figure(figsize=(10, 8))  # Adjust the figure size as per your preference
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    
    # Increase title size
    plt.title(title, fontweight='bold', fontsize=10 * text_size_factor, pad=10)
    plt.colorbar()

    # Generate tick marks
    tick_marks = np.arange(len(class_names)) if class_names is not None else np.arange(len(conf_matrix))
    plt.xticks(tick_marks, class_names, fontsize=10 * text_size_factor)
    plt.yticks(tick_marks, class_names, fontsize=10 * text_size_factor)
    
    # Increase axis label sizes
    plt.xlabel('Predicted label', fontsize=12 * text_size_factor)
    plt.ylabel('True label', fontsize=12 * text_size_factor)

    # Add counts and percentages in each box with increased text size
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            percentage = conf_matrix_norm[i, j] * 100
            count = int(conf_matrix[i, j])
            text_color = 'black' if percentage < 80 else 'white'
            plt.text(j, i, "{:.2f}%\n({})".format(percentage, count),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color=text_color,
                     fontsize=10 * text_size_factor)  # Increase size of box text

    if save:
        confusion_matrices_path = pathmaster.confusion_matrices_path()
        confusion_matrices_path = os.path.join(confusion_matrices_path, 'confusion_matrix_' + pathmaster.file_tag + '.jpg')
        
        # Save the figure with tight bounding box
        plt.savefig(confusion_matrices_path, dpi=200, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        
def train_val_loss(losslist_train, losslist_val, title='Training and Validation Loss', save=False, pathmaster=None):
    title = title + ' (' + pathmaster.file_tag + ')'
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(losslist_train)), losslist_train, label='training')
    plt.plot(range(len(losslist_val)), losslist_val, label='validation')
    plt.legend()
    plt.title(title, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    if save:
        loss_curves_path = pathmaster.loss_curves_path()
        loss_curves_path = os.path.join(loss_curves_path, 'loss_curve_' + pathmaster.file_tag + '.jpg')
        plt.savefig(loss_curves_path, dpi=150)
        
    # plt.show()
    
def accuracy_curves(Y_true_train, Y_true_val, Y_pred_train, Y_pred_val, title='Training and Validation Accuracy', save=False, pathmaster=None):
    accuracy_list_train = []
    accuracy_list_val = []
    epochs_train = range(len(Y_pred_train))
    epochs_val = range(len(Y_pred_val))
    
    for predictions in Y_pred_train:
        accuracy = accuracy_score(Y_true_train, predictions)
        accuracy_list_train.append(accuracy)
    for predictions in Y_pred_val:
        accuracy = accuracy_score(Y_true_val, predictions)
        accuracy_list_val.append(accuracy)
        
    title = title + ' (' + pathmaster.file_tag + ')'
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_train, accuracy_list_train, label='training')
    plt.plot(epochs_val, accuracy_list_val, label='validation')
    plt.legend()
    plt.title(title, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    if save:
        accuracy_curves_path = pathmaster.accuracy_curves_path()
        accuracy_curves_path = os.path.join(accuracy_curves_path, 'accuracy_curve_' + pathmaster.file_tag + '.jpg')
        plt.savefig(accuracy_curves_path, dpi=150)
        
def save_segment_names(segment_names, pathmaster):
    segment_names_path = pathmaster.segment_names_path()
    segment_names_path = os.path.join(segment_names_path, 'segment_names_' + pathmaster.file_tag + '.csv')
    
    # Using PyArrow
    segment_names_table = pa.Table.from_arrays([segment_names], names=['segment_names'])
    csv.write_csv(segment_names_table, segment_names_path)
    
def save_output_file(segment_names, predictions, ground_truths, pathmaster):
    output_file_path = pathmaster.output_file_path()
    output_file_path = os.path.join(output_file_path, 'output_files_' + pathmaster.file_tag + '.csv')
    
    output_file = pd.DataFrame()
    output_file['segment_name'] = segment_names
    output_file['prediction'] = predictions
    output_file['ground_truth'] = ground_truths
    
    output_file.to_csv(output_file_path, index=False)