# UConnThesis
Darren, last edit 03/23/2025.

This branch contains the pretrained DenseNet models, source code, and data loading documentation for Darren Chen's Honors Undergraduate Thesis, titled **"Application of Deep Learning and Data Balancing Methods for Multiclass Cardiac Rhythm Detection and Classification Using Real-World Smartwatch Photoplethysmography"** at the **University of Connecticut**.  

Keywords:
- PPG: photoplethysmography
- AF: atrial fibrilaltion
- PAC/PVC or PAC_PVC: premature atrial and ventricular contractions
- NSR: normal sinus rhythm

The study focuses on the **Pulsewatch dataset** to develop and evaluate deep learning models for detecting and classifying various cardiac rhythms using real-world smartwatch PPG data. This work strongly builds upon the DenseNet architecture and extends its application to cardiac rhythm classification using real-world smartwatch PPG data.  

- DenseNet Repository: [DenseNet by Zhuang Liu](https://github.com/liuzhuang13/DenseNet)  
- For reference, see:
    - G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2017.

This project also builds upon and extends previous research on Pulsewatch. For more information, please refer to the Pulsewatch repository maintained by my postdoc mentor: [PulsewatchRelease](https://github.com/Cassey2016/PulsewatchRelease.git). 

## Training, Validating, and/or Testing Model
- Go to the `README.md` file in the `main` folder.
- Go to `main.py` in the `main` folder.

## Optimizing Model
- Go to the `README.md` file in the `main` folder.
- Go to `main_random_search.py` in the `main` folder.

## Setting Up Paths
- Go to the `README.md` file in the `utils` folder.
- Go to `pathmaster.py` in the `utils` folder.

## Generate Time-Frequency Spectrogram (TFS) Images
- Go to the `README.md` file in the `generate_data` folder.
- Go to `generate_tfs.ipynb` in the `generate_data` folder.

## Artificially Upsample Dataset of TFS Images
- Go to the `README.md` file in the `generate_data` folder.
- Go to `smote_accelerated.py` in the `generate_data` folder.

## General Directory Setup
Please note that file names are generalized and that not all directories are shown.
```
.
├── densenet-for-smartwatch-ppg
    ├── generate_data
    ├── main
    ├── models [models_path in pathmaster.models_path()]
    ├── pretrained_models
    ├── utils
├── outputs [pathmaster.root_saves_path]
    ├── focus_1
        ├── file_tag_1
            ├── accuracy_curves
            ├── checkpoints
            ├── classification_reports
            ├── classification_reports_imbalanced
            ├── confusion_matrices
            ├── hyperparameters
            ├── labels
            ├── loss_curves
            ├── losslists
            ├── mean_roc_curves
            ├── metrics
            ├── output_files
            ├── prediction_proba
            ├── predictions
            ├── roc_curves
            ├── runtime_lists
            ├── segment_names
        ├── file_tag_2
        ├── ...
    ├── focus_2
    ├── ...
├── NIH_Pulsewatch [data_root_path in pathmaster.data_paths()]
    ├── 1d_ppg_csv
        ├── 001
            ├── ppg_segment_1.csv
            ├── ppg_segment_2.csv
            ├── ...
        ├── 002
        ├── ...
    ├── Ground_Truths [labels_path in pathmaster.data_paths()]
        ├── 001_ground_truth.csv
        ├── 002_ground_truth.csv
        ├── ...
    ├── TFS_pt
        ├── 128x128_float16 [data_path in pathmaster.data_paths()]
            ├── 001
                ├── ppg_tfs_1.pt
                ├── ppg_tfs_2.pt
                ├── ...
            ├── 002
                ├── ...
        ├── combined_dataset
            ├── multiclass [combination_path in pathmaster.combination_path()]
        ├── SMOTE
            ├── holdout_60_10_30 [smote_path in pathmaster.smote_path()]
        ├── Borderline_SMOTE
            ├── holdout_60_10_30 [smote_path in pathmaster.smote_path()]
        ├── ADASYN
            ├── holdout_60_10_30 [smote_path in pathmaster.smote_path()]
        ├── ...
    ├── labels_summary.csv [summary_path in pathmaster.summary_path()]
├── Public_Database
    ├── MIMIC_III
        ├── segment_names_and_ground_truth_labels.csv [labels_path in pathmaster.mimic3_paths()]
        ├── test_tfs_float16_pt [data_path in pathmaster.mimic3_paths()]
            ├── 001
                ├── ppg_tfs_1.pt
                ├── ppg_tfs_2.pt
                ├── ...
            ├── 002
                ├── ...
            ├── ...
    ├── Simband
        ├── segment_names_and_ground_truth_labels.csv [labels_path in pathmaster.simband_paths()]
        ├── test_tfs_float16_pt [data_path in pathmaster.simband_paths()]
            ├── 001
                ├── ppg_tfs_1.pt
                ├── ppg_tfs_2.pt
                ├── ...
            ├── 002
                ├── ...
            ├── ...
    ├── ...
```
