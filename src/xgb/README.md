# XGBoost Baseline

This directory contains an XGBoost regression baseline for precipitation prediction.

## Structure

- `model.py`: XGBoost model wrapper class
- `train.py`: Training script for XGBoost models
- `eval.py`: Evaluation script for generating predictions
- `configs.json`: Configuration file for experiments and hyperparameters

## Usage

### Training

To train XGBoost models, run from this directory:

```bash
python train.py
```

This will train all models specified in the `runs` section of `configs.json`.

### Evaluation

To evaluate trained models on a validation or test set:

```bash
python eval.py --split norm_valid  # Evaluate on validation set
python eval.py --split norm_test   # Evaluate on test set
```

Predictions will be saved to the results directory specified in `configs.json`.

## Configuration

The `configs.json` file contains:

- **paths**: Directories for data, models, and results
- **dataparams**: Variable names (target and land fraction)
- **experiments**: Input variable configurations
- **trainparams**: XGBoost hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- **evalparams**: Batch size for inference
- **runs**: Specific model runs combining experiments with different objectives

## XGBoost Hyperparameters

Default hyperparameters in `configs.json`:

- `n_estimators`: 500 (number of boosting rounds)
- `max_depth`: 8 (maximum tree depth)
- `learning_rate`: 0.1 (step size shrinkage)
- `subsample`: 0.8 (fraction of samples per tree)
- `colsample_bytree`: 0.8 (fraction of features per tree)
- `tree_method`: 'hist' (histogram-based tree construction)
- `early_stopping_rounds`: 50 (stop if no validation improvement)
- `objective`: 'reg:squarederror' or 'reg:absoluteerror' (MSE or MAE loss)

## Notes

- Uses the same normalized data as the NN models
- Supports both MSE and MAE objectives
- Optional land fraction as an additional input feature
- Predictions are denormalized back to mm/hr units
- Models are saved in JSON format for portability
