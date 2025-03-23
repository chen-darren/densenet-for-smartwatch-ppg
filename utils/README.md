# Configuring Paths for PathMaster
Darren, last edit 03/23/2025.

## Adjust All Specific Paths
Only change specific string paths.
- Note that there are diffierent paths depending on the type of device or drive used. Be sure to change EVERYTHING!

### `__init__()`
`self.root_saves_path`
- Root directory for where the results of each run of each focus should be saved.
- Note that this directory will branch into the `focus` subdirectories

### `data_paths()` (for Pulsewatch dataset)
`data_root_path`
- Root directory for where the data (i.e. TFS images) are stored
- Note that this directory will branch into the `format_path` subdirectories (i.e. 'TFS_pt') which then branch into `img_res` subdirectories (128x128_float16).

`labels_root_path`
- Root directory for where the labels are stored
    - Everything works if the same as `data_root_path`. If different, somethings may not work...
- Note that this directory will branch into the `Ground_Truth` subdirectory which contains individual CSVs for each UID, each CSV containins the segment names and corresponding labels of each segment for that UID.

### `combination_path()`
`root_path`
- Root directory for where the data of combined datasets should be saved.
    - Everything works if the same as `data_root_path`. If different, somethings may not work...
- Note that this directory will branch into the `format_path` subdirectories (i.e. 'TFS_pt') which then branch into the `combination` subdirectories (i.e. 'combined_dataset', 'pulsewatch_simband', 'simband_mimic3') which then branch into the `split` subdirectory (i.e. 'multiclass').

### `deapbeat_paths()`
`root_path`
- Root directory for where the data (i.e. TFS images) are stored.
- Note that this directory will branch into the `format_path` subdirectories (i.e. 'tfs_float16_pt').

`labels_path`
- Path to the CSV containing the segment names and labels for the dataset.

### `mimic3_paths()`
`root_path`
- Root directory for where the data (i.e. TFS images) are stored.
- Note that this directory will branch into the `format_path` subdirectories (i.e. 'tfs_float16_pt').

`labels_path`
- Path to the CSV containing the segment names and labels for the dataset.

### `simband_paths()`
`root_path`
- Root directory for where the data (i.e. TFS images) are stored.
- Note that this directory will branch into the `format_path` subdirectories (i.e. 'tfs_float16_pt').

`labels_path`
- Path to the CSV containing the segment names and labels for the dataset.

### `summary_path()`
`summary_path`
- Path to the CSV containing the labels summary of the dataset (Pulsewatch)

### `model_path()`
`model_path`
- Path to the directory containing the model script (i.e. densenet_configurable)