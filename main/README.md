# Multiclass Classifcation of NSR, AF, and PAC/PVC Using a DenseNet Model 
Darren, last edit 03/23/2025.

## Ensure Correct Paths for Your Device
Open `pathmaster.py` in the `utils` subdirectory
- Saves paths
    - Go to the the initialization function of PathMaster: `__init__(args)`
    - Set the correct saves paths for the each of the relevant platforms (Linux, HPC, and Windows (else))
- Data paths
    - Go to `data_paths(args)`
    - Set the correct `base_path` and `labels_base_path` for each platform and ensure that the data_paths exist for each data format
- Other data paths
    - If you are using other databases, ensure that the respective paths functions are correct
- Create all relevant saves directories based on the specific saves paths functions
- See README at `densenet-for-smartwatch-ppg\utils\README.md` for more information about paths and PathMaster.

## `main.py`
Used for both training and testing of the DenseNet model.

### Run Parameters
Set the desired run parameters

#### Device and Drives for the Dataset
`is_linux`
- Boolean for whether Linux is being used or not.

`is_hpc`
- Boolean for whether HPC is being used or not.

`is_internal`
- Boolean for whether an internal storage device is being used or not.

`is_external`
- Boolean for whether an external storage device is being used or not.

#### Input Type
`is_tfs`
- Boolean for whether TFS is being used, if False, density Poincare plots are used.
- Note that the code was not tested thoroughly with density Poincare plots.

`binary`
- Boolean for binary classification (non-AF vs AF)

#### Run Identifiers
`focus`
- A string identifier for the broad goal of a group of runs (i.e. 'random_search', 'thesis_results_final_pulsewatch_multiclass', 'thesis_results_final_pulsewatch_binary', etc.).

`file_tag`
- A string identifier for the specific run under a single focus (i.e. 'random_search_0', 'random_search_1', 'Proposed_Model', 'Pulsewatch', etc.).

`split`
- String describing how the data was split ('holdout_60_10_30')

#### Combined Datasets
`combination`
- String describing the type of combined dataset (i.e. 'combined_dataset', 'pulsewatch_simband', 'simband_mimic3').

`split`
- String describing how the data was split ('multiclass')

#### Database
If you are using another database (e.g. DeepBeat, Simband, or MIMIC III)...

`database`
- String text of common ways to name the databases

#### Data and Image Settings
`img_res`
- String describing the resolution (and potentially precision) of the images used as input.
- Proposed methodology used '128x128_float16'.

`img_channels`
- Int number of input image channels.
- Proposed methodology used 1 channel TFS images.

`img_size`
- Int resolution of the input image channels.
- Proposed methodology used 128 (128x128).

`downsample`
- Int of the new downsampled resolution.
- None if no downsampling desired.

`standardize`
- Boolean for whether standardization is wanted.

`data_type`
- The PyTorch datatype to convert the data into upon loading.
- Proposed methodology used torch.float32.
    - May run into issues if use torch.float16.

#### Other
`model_type`
- The dtype to initially load the model with.
- Keep at torch.float32 as the inital dtype as mixed precision is being used for actual running of the model.

`save`
- Boolean describing whether or not the model should save its results

`data_format`
- 'csv' or 'pt'...strongly suggest using 'pt' for fastest model

`n_epochs`
- Int value for the number of epochs to run.

`batch_size`
- Int value for the size of each batch.

### UID Split
All split functions return three lists of UIDs for the training, validation, and testing sets

`dataloader.split_uids(pathmaster)`
- Original split

`dataloader.split_uids_60_10_30(pathmaster)`
- Holdout 60:10:30 with respect to the number of PAC/PVC segments for all classes (undersampled NSR and AF).

`dataloader.split_uids_60_10_30_v2(pathmaster)`
- Holdout 60:10:30 with respect to number of segments and subjects for all classes (no undersampling).
- Used to manually calculate class weights.

`dataloader.split_uids_60_10_30_balanced(pathmaster)`
- Holdout 60:10:30 with respect to number of segments and subjects for all classes, but the training set has the majority classes downsampled to achieve a balanced training set.

`dataloader.split_uids_60_10_30_noPACPVC(pathmaster)`
- Holdout 60:10:30 split with respect to the number of segments and subjects with respect to NSR and AF (no PAC/PVC).

### Preprocess Data and Prepare Dataloaders
`dataloader.preprocess_data(args)`
- Returns holdout training, validation, and test loaders.

### Preprocess Data and Prepare Dataloaders for Other Databases
`dataloader_database.preprocess_data(args)`
- Returns test loader.

### Preprocess Data and Prepare Dataloaders for Other Databases with 60:10:30 Split
`dataloader_database.preprocess_data_split(args)`
- Returns holdout training, validation, and test loaders.

### Preprocess Data and Prepare Dataloaders for Combined Dataset
`dataloader_database.preprocess_data_split(args)`
- Returns holdout training and validation loaders.

### Model Hyperparameters
Set desired hyperparameters in `config`.
- `num_layers_per_dense`: number of convolutional layers per dense block
- `growth_rate`: number of feature maps to add for each layer
- `compression`: factor to reduce the number of feature maps following each transition layer
- `drop_rate`: rate of dropout
- `class_weights`: weights for the classes for the loss function
    - Make sure the number of weights matches the number of classes (3 for multiclass and 2 for binary)
- `learning_rate`: learning rate for the Adam optimizer
- `lambda_l1`: lambda for L1 regularization
- `activation`: activation function
    - ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Softplus, Softshrink, Softmin, LogSoftmax, ELU, PreLU, ReLU6, RReLU, CELU, SELU, GELU, and SiLU

### Training, Validation, and Testing
If you want to train, validate, and test, simply run the Python script after setting all of the configurables and ensuring all paths are correct.

### Only Training and Validation or Only Testing
If you want to only perform training and validation:
- Comment out entire testing code block from `# # Run best model =======...` (Line ~179) to `print('\nTesting took %.2f' % time_passed, 'seconds')` (Line ~189)

If you want to only perform testing:
- Comment out entire training and validation code block from `# # Training and validationn =======...` (Line ~155) to `print('\nTraining and validation took %.2f' % time_passed, 'minutes')` (Line ~177)
- Note that model loading for testing is based purely on PathMaster and its `focus` and `file_tag`: the model weights located at `os.path.join(pathmaster.root_saves_path, focus, file_tag, 'checkpoints)` with hyperparameters at `os.path.join(pathmaster.root_saves_path, focus, file_tag, 'hyperparameters)` is loaded
    - If the model weights and hyperparameters do not match, there will be an error!

## `main_random_search.py`
Used for optimization of hyperparameters through a random search over a defined hyperparameter space.

### Run Parameters in `main()`
Set the desired run parameters in `main()`

#### Device and Drives for the Dataset
`is_linux`
- Boolean for whether Linux is being used or not.

`is_hpc`
- Boolean for whether HPC is being used or not.

`is_internal`
- Boolean for whether an internal storage device is being used or not.

`is_external`
- Boolean for whether an external storage device is being used or not.

#### Input Type
`is_tfs`
- Boolean for whether TFS is being used, if False, density Poincare plots are used.
- Note that the code was not tested thoroughly with density Poincare plots.

`binary`
- Boolean for binary classification (non-AF vs AF)

#### Run Identifiers
`focus`
- A string identifier for the broad goal of a group of runs (i.e. 'random_search', 'thesis_results_final_pulsewatch_multiclass', 'thesis_results_final_pulsewatch_binary', etc.).

`file_tag`
- Does not matter.
- Will automatically be overwritten with 'random_search_0', 'random_search_1', 'random_search_2', ...

#### Data and Image Settings
`img_res`
- String describing the resolution (and potentially precision) of the images used as input.
- Proposed methodology used '128x128_float16'.

`img_channels`
- Int number of input image channels.
- Proposed methodology used 1 channel TFS images.

`dataset`
- String identifier for which dataset to use for training and validation (i.e. 'pulsewatch', 'combined')

#### Hyperparameter Search Space
`config`: dictionary containing each hyperparameter to be optimized (key) and its search space as a list (value)
```
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
```

### Run Parameters in `load_data()`
Set the desired run parameters in `load_data()`

#### Data and Image Settings
`img_channels`
- Int number of input image channels.
- Proposed methodology used 1 channel TFS images.
- Ensure it matches `img_channels` in `main()`

`img_size`
- Int resolution of the input image channels.
- Proposed methodology used 128 (128x128).

`downsample`
- Int of the new downsampled resolution.
- None if no downsampling desired.

`standardize`
- Boolean for whether standardization is wanted.

`data_type`
- The PyTorch datatype to convert the data into upon loading.
- Proposed methodology used torch.float32.
    - May run into issues if use torch.float16.

#### Other
`data_format`
- 'csv' or 'pt'...strongly suggest using 'pt' for fastest model

`batch_size`
- Int value for the size of each batch.

### Run Parameters in `train_DenseNet()`
Set the desired run parameters in `train_DenseNet()`

#### Data and Image Settings
`img_channels`
- Int number of input image channels.
- Proposed methodology used 1 channel TFS images.
- Ensure it matches `img_channels` in `main()`

### Run Arguments for `main()`
`n_iter`
- Int number of iterations to run the random search.
`n_epochs`
- Int max number of epochs to run for each iteration.
`save`
- Boolean describing whether or not the model should save its results.
`start_iter`
- Int number noting the iteration to start on.
- Used to resume the random search.

### Perform Random Search
After setting all of the parameters and arguments and ensuring the paths are correct, run the script!