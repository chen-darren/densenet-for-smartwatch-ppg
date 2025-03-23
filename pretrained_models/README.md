# Pretrained Models
Darren, last edit 03/23/2025.

## Checkpoints
Contains the `checkpoint_model_name.pt` files of each pretrained model.

## Hyperparameters
Contains the `hyperparameters_model_name.csv` files of each pretrained model.
- Note that all hyperparameters of all pretrained models, other than the weights, are the same. The weights are specific to either the multiclass (NSR, AF, PAC/PVC) or binary (non-AF: NSR + PAC/PVC, AF) models.

## Pretrained Models
### Proposed_Model_Multiclass
Trained and validated on the holdout 60:10:30 split of Pulsewatch data for multiclass (NSR, AF, PAC/PVC) classification.

### Proposed_Model_Combined Multiclass
Trained and validated on the combined dataset of holdout 60:10:30 splits of the Pulsewatch, Simband, and MIMIC-III datasets for multiclass (NSR, AF, PAC/PVC) classification.