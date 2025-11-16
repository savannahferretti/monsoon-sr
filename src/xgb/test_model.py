#!/usr/bin/env python

"""
Simple test script to verify XGBoost model implementation works correctly.
"""

import os
import tempfile
import numpy as np
from model import XGBModel

def test_model_basic():
    """Test basic model training and prediction"""
    print("Testing XGBoost model implementation...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    X_train = np.random.randn(n_samples, n_features)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(n_samples) * 0.1

    X_valid = np.random.randn(200, n_features)
    y_valid = X_valid[:, 0] * 2 + X_valid[:, 1] * 3 + np.random.randn(200) * 0.1

    # Test 1: Model initialization
    print("  [1/6] Testing model initialization...")
    model = XGBModel(
        objective='reg:squarederror',
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        tree_method='hist'
    )
    assert model.model is None, "Model should be None before fitting"
    print("      ✓ Model initialization successful")

    # Test 2: Model training
    print("  [2/6] Testing model training...")
    model.fit(X_train, y_train, X_valid, y_valid, early_stopping_rounds=10, verbose=False)
    assert model.model is not None, "Model should not be None after fitting"
    print("      ✓ Model training successful")

    # Test 3: Prediction
    print("  [3/6] Testing prediction...")
    y_pred = model.predict(X_valid)
    assert y_pred.shape == (200,), f"Expected shape (200,), got {y_pred.shape}"
    print(f"      ✓ Prediction successful (shape: {y_pred.shape})")

    # Test 4: Model saving and loading
    print("  [4/6] Testing model save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_model.json')
        model.save(filepath)
        assert os.path.exists(filepath), "Model file should exist after saving"

        # Load model
        new_model = XGBModel(
            objective='reg:squarederror',
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            tree_method='hist'
        )
        new_model.load(filepath)
        y_pred_loaded = new_model.predict(X_valid)

        # Predictions should be identical
        assert np.allclose(y_pred, y_pred_loaded), "Predictions should match after loading"
        print("      ✓ Model save/load successful")

    # Test 5: Different objectives
    print("  [5/6] Testing MAE objective...")
    model_mae = XGBModel(
        objective='reg:absoluteerror',
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1
    )
    model_mae.fit(X_train, y_train, X_valid, y_valid, verbose=False)
    y_pred_mae = model_mae.predict(X_valid)
    assert y_pred_mae.shape == (200,), "MAE prediction shape should be correct"
    print("      ✓ MAE objective successful")

    # Test 6: Error handling
    print("  [6/6] Testing error handling...")
    untrained_model = XGBModel()
    try:
        untrained_model.predict(X_valid)
        assert False, "Should raise RuntimeError for untrained model"
    except RuntimeError:
        print("      ✓ Error handling successful")

    print("\n✅ All tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_model_basic()
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        raise
