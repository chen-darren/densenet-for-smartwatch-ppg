# Generating Time-Frequency Spectrogram (TFS) Images from 30-Second PPG Segments
Darren, last edit 03/22/2025.

## Requirements
- All 30-second PPG segments must be in a 1D CSV format as a single column vector.
- Every database to be converted must have a `menu`: two-column CSV with headers with the first column being `segment_names` and the second column being `labels` containing the names and labels of each 1D 30-second PPG segment to be converted into TFSs.

## Generate TFS of Any Database
- Open `generate_tfs.ipynb`.
- Recommend to change all paths in `create_paths` to your local path that stores the respective data.
  - input_path = `r'\path\to\directory\containing\1d_ppg_segments_as_csv'`
  - menu_path = `r'\path\to\directory\containing\menu_for_database'`
  - output_path_fig = `r'\path\to\directory\where\tfs_plots\should\be\saved'`
  - output_path_csv = `r'\path\to\directory\where\tfs_csv\should\be\saved'`
  - filename_menu = `'name_of_menu_for_database'`
- Running the script
  - If resuming a conversion, set the correct starting index in `my_main_instance.set_MyMain`.
  - Specify the desired database in `my_main_instance.set_MyMain`.
    - Recommend to change database names to your personal database names.
  - Parameters of `my_main_instance.my_main_func`
    - Set the sampling rate of the input 1D PPG by setting `sample_rate`.
    - To resample, set the target frequency by setting `resample`, otherwise set to `None`.
    - To filter, set `filter` to `True` (or `False`, if you don't want to filter).
    - To save TFS plots, set `plot` to `True` or `False` otherwise.
- Note: To change TFS size (default is 128x128), set `newsize` in `my_main_func` to desired size.
- Execute all cells to convert the 1D PPG CSV into TFS CSV.

# Artifically Increasing Dataset with SMOTE
Darren, last edit 03/22/2025.

## Requirements
- TFS plots of 30-second PPG segments located in the appropriate directories: see `README.md in densenet-for-smartwatch-ppg` for details about directory specifications.
- Must use a PC without a GPU: `torch.cuda.is_available() must be False`

## Using SMOTE
- Open `smote_acclerated.py`
- Set configurable settings for the run in `main()`.
  - `smote_type`: The type of SMOTE to use.
    - 'SMOTE', 'Borderline_SMOTE', or 'ADASYN' (case-sensitive)
  - `split`: The identifier used for the split that is being SMOTE'd.
    - i.e. 'holdout_60_10_30'
  - `groups`: List containing the groups to be SMOTE'd.
    - Typically use ['train', 'validate', 'test'].
  - `is_linux`: Boolean for whether Linux is being used or not.
  - `is_hpc`: Boolean for whether HPC is being used or not.
  - `is_internal`: Boolean for whether an internal storage device is being used or not.
  - `is_external`: Boolean for whether an external storage device is being used or not.
  - `is_tfs`: Boolean for whether TFS is being used, if False, density Poincare plots are used.
  - `img_res`: String describing the resolution (and potentially precision) of the images used as input.
    - Proposed methodology used '128x128_float16'.
  - `img_channels`: Int number of input image channels.
    - Proposed methodology used 1 channel TFS images.
  - `img_size`: Int resolution of the input image channels (assuming square).
    - Proposed methodology used 128.
  - `downsample`: Int of the new downsampled resolution.
    - None if no downsampling desired.
  - `standardize`: Boolean for whether standardization is wanted.
  - `data_type`: The PyTorch datatype to convert the data into upon loading.
    - Proposed methodology used torch.float32.
    - May run into issues if use torch.float16.
  - `data_format`: Format of TFS images.
    - 'csv' or 'pt'...strongly suggest using 'pt' for fastest model
  - `batch_size`: The batch size to be used
- Run `smote_acclerated.py` to use SMOTE