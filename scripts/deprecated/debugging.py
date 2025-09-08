#!/usr/bin/env python

"""
debugging.py

Runs all sanity checks for each experiment in nn.CONFIGS:
  1) Mean baseline (train, valid)
  2) Linear regression baseline (train, valid)
  3) NN untrained scores (train, valid)  [mini-batch inference to avoid CUDA OOM]
  4) Train NN (opens a disabled wandb run so nn.NNMODEL.fit can log safely)
  5) NN trained scores (train, valid, test) [mini-batch inference]

Outputs a console table + saves debug_results.csv and debug_results.json
Usage:
  python debugging.py
  # optional overrides:
  python debugging.py --epochs 10 --lr 3e-4 --patience 6 --batchsize 32768
"""

import os
import json
import csv
import argparse
import numpy as np
import torch

# --- import from your local nn.py (same folder) ---
from nn_new import (
    load,        # loads X/y splits + normparams
    NNMODEL,     # your model class
    CONFIGS,     # list of experiments
    NORMTARGET, LOG1PTARGET,
)

# ---------------- helpers (self-contained; no dependency on nn.invert_normalization) ----------------

def invert_normalization(y_norm, normparams):
    """Invert the target normalization that nn.load() applied."""
    if normparams is None:
        return y_norm
    y = y_norm * normparams['std'] + normparams['mean']
    if normparams.get('log1p', False):
        y = np.expm1(y)
    return y

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def r2_score(y_true, y_pred):
    y_true = _to_numpy(y_true).astype(np.float64)
    y_pred = _to_numpy(y_pred).astype(np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def mse(y_true, y_pred):
    y_true = _to_numpy(y_true); y_pred = _to_numpy(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.mean((y_true[mask] - y_pred[mask])**2)

def mae(y_true, y_pred):
    y_true = _to_numpy(y_true); y_pred = _to_numpy(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

def to_original(y_norm, normparams):
    return invert_normalization(_to_numpy(y_norm), normparams)

def metrics_on_original_scale(y_true_norm, y_pred_norm, normparams):
    y_true = to_original(y_true_norm, normparams)
    y_pred = to_original(y_pred_norm, normparams)
    return {'R2': r2_score(y_true, y_pred),
            'MSE': mse(y_true, y_pred),
            'MAE': mae(y_true, y_pred)}

def constant_mean_baseline(y_train_norm, y_eval_norm, normparams):
    """Predict mean(y_train) everywhere; evaluate on ORIGINAL scale."""
    y_train_orig = to_original(y_train_norm, normparams)
    const = np.mean(y_train_orig)
    y_eval_orig = to_original(y_eval_norm, normparams)
    y_pred_orig = np.full_like(y_eval_orig, const)
    return {'R2': r2_score(y_eval_orig, y_pred_orig),
            'MSE': mse(y_eval_orig, y_pred_orig),
            'MAE': mae(y_eval_orig, y_pred_orig)}

def linear_regression_baseline(
    X_train, y_train_norm, X_eval, y_eval_norm, normparams,
    ridge=1e-6, max_samples=int(5e6), random_state=0
):
    """Closed-form ridge on ORIGINAL y; evaluate on ORIGINAL y."""
    rng = np.random.default_rng(random_state)
    X_tr = _to_numpy(X_train)
    X_ev = _to_numpy(X_eval)
    y_tr_orig = to_original(y_train_norm, normparams)
    y_ev_orig = to_original(y_eval_norm, normparams)

    n = X_tr.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_tr_fit = X_tr[idx]
        y_tr_fit = y_tr_orig[idx]
    else:
        X_tr_fit = X_tr
        y_tr_fit = y_tr_orig

    ones_tr = np.ones((X_tr_fit.shape[0], 1), dtype=X_tr_fit.dtype)
    ones_ev = np.ones((X_ev.shape[0], 1),    dtype=X_ev.dtype)
    Xb_tr = np.concatenate([X_tr_fit, ones_tr], axis=1)
    Xb_ev = np.concatenate([X_ev,      ones_ev], axis=1)

    d = Xb_tr.shape[1]
    XtX = Xb_tr.T @ Xb_tr
    XtX += ridge * np.eye(d, dtype=XtX.dtype)
    Xty = Xb_tr.T @ y_tr_fit
    w = np.linalg.solve(XtX, Xty)

    y_pred_ev = Xb_ev @ w
    return {'R2': r2_score(y_ev_orig, y_pred_ev),
            'MSE': mse(y_ev_orig, y_pred_ev),
            'MAE': mae(y_ev_orig, y_pred_ev)}

# ---------------- OOM-safe inference ----------------

def predict_in_batches(model, X, device, batch_size=65536, cpu_fallback=True):
    """
    Run model(X) in chunks to limit memory. Forces eval() so BatchNorm uses running stats.
    """
    model.model.eval()
    preds = []
    try:
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = X[i:i+batch_size].to(device, non_blocking=True)
                yb = model.model(xb).squeeze().cpu().numpy()
                preds.append(yb)
        return np.concatenate(preds, axis=0)
    except torch.cuda.OutOfMemoryError:
        if cpu_fallback and device.type == 'cuda':
            torch.cuda.empty_cache()
            model.model.cpu()
            preds = []
            with torch.no_grad():
                for i in range(0, X.shape[0], batch_size):
                    xb = X[i:i+batch_size].cpu()
                    yb = model.model(xb).squeeze().numpy()
                    preds.append(yb)
            model.model.to(device)
            return np.concatenate(preds, axis=0)
        else:
            raise

# ---------------- main experiment runner ----------------

def run_one_experiment(name, inputvars, filename, model_overrides):
    # 1) Load data and normparams
    Xtr, Xva, Xte, ytr, yva, yte, normparams = load(inputvars, filename)

    # 2) Baselines (orig scale)
    mean_tr = constant_mean_baseline(ytr, ytr, normparams)
    mean_va = constant_mean_baseline(ytr, yva, normparams)
    lin_tr  = linear_regression_baseline(Xtr, ytr, Xtr, ytr, normparams)
    lin_va  = linear_regression_baseline(Xtr, ytr, Xva, yva, normparams)

    # 3) NN untrained (orig scale; OOM-safe)
    model = NNMODEL(Xtr.shape[1], **model_overrides)
    device = model.device
    ypred_tr_init = predict_in_batches(model, Xtr, device, batch_size=65536)
    ypred_va_init = predict_in_batches(model, Xva, device, batch_size=65536)
    nn_init_tr = metrics_on_original_scale(ytr, ypred_tr_init, normparams)
    nn_init_va = metrics_on_original_scale(yva, ypred_va_init, normparams)

    # 4) Train NN (open a disabled wandb run so nn.fit() can log)
    try:
        import wandb
        wandb.init(mode="disabled")
    except Exception:
        wandb = None
    model.fit(Xtr, Xva, ytr, yva)
    if wandb is not None:
        try: wandb.finish()
        except Exception: pass

    # 5) NN trained (orig scale; OOM-safe)
    ypred_tr = predict_in_batches(model, Xtr, device, batch_size=65536)
    ypred_va = predict_in_batches(model, Xva, device, batch_size=65536)
    ypred_te = predict_in_batches(model, Xte, device, batch_size=65536)
    nn_tr = metrics_on_original_scale(ytr, ypred_tr, normparams)
    nn_va = metrics_on_original_scale(yva, ypred_va, normparams)
    nn_te = metrics_on_original_scale(yte, ypred_te, normparams)

    return {
        "name": name,
        "inputvars": inputvars,
        "target_normalized": bool(NORMTARGET),
        "target_log1p": bool(LOG1PTARGET),

        "mean_train_R2": mean_tr["R2"],
        "mean_valid_R2": mean_va["R2"],
        "lin_train_R2":  lin_tr["R2"],
        "lin_valid_R2":  lin_va["R2"],

        "nn_untrained_train_R2": nn_init_tr["R2"],
        "nn_untrained_valid_R2": nn_init_va["R2"],

        "nn_trained_train_R2": nn_tr["R2"],
        "nn_trained_valid_R2": nn_va["R2"],
        "nn_trained_test_R2":  nn_te["R2"],

        # extras in physical units
        "mean_valid_MAE": mean_va["MAE"],
        "lin_valid_MAE":  lin_va["MAE"],
        "nn_untrained_valid_MAE": nn_init_va["MAE"],
        "nn_trained_valid_MAE":   nn_va["MAE"],
        "nn_trained_test_MAE":    nn_te["MAE"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="ml_data_subset.h5", help="HDF5 file from split.py")
    ap.add_argument("--outdir", default=".", help="Where to save CSV/JSON")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batchsize", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    args = ap.parse_args()

    model_overrides = {}
    if args.epochs is not None:    model_overrides["epochs"] = args.epochs
    if args.batchsize is not None: model_overrides["batchsize"] = args.batchsize
    if args.lr is not None:        model_overrides["learningrate"] = args.lr
    if args.patience is not None:  model_overrides["patience"] = args.patience

    os.makedirs(args.outdir, exist_ok=True)
    print("\n=== Running debug checks ===")
    print(f"Data file: {args.file}")
    print(f"Target normalized: {bool(NORMTARGET)} | log1p: {bool(LOG1PTARGET)}\n")

    header = (
        "name",
        "mean_tr_R2","mean_va_R2",
        "lin_tr_R2","lin_va_R2",
        "nn_init_tr_R2","nn_init_va_R2",
        "nn_tr_tr_R2","nn_tr_va_R2","nn_tr_te_R2",
    )
    print("{:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>12} {:>12} | {:>10} {:>10} {:>10}".format(*header))
    print("-"*114)

    results = []
    for cfg in CONFIGS:
        res = run_one_experiment(cfg["name"], cfg["inputvars"], args.file, model_overrides)
        results.append(res)
        row = (
            res["name"],
            f"{res['mean_train_R2']:.3f}", f"{res['mean_valid_R2']:.3f}",
            f"{res['lin_train_R2']:.3f}",  f"{res['lin_valid_R2']:.3f}",
            f"{res['nn_untrained_train_R2']:.3f}", f"{res['nn_untrained_valid_R2']:.3f}",
            f"{res['nn_trained_train_R2']:.3f}",   f"{res['nn_trained_valid_R2']:.3f}", f"{res['nn_trained_test_R2']:.3f}",
        )
        print("{:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>12} {:>12} | {:>10} {:>10} {:>10}".format(*row))

    # Save CSV
    csv_path = os.path.join(args.outdir, "debug_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save JSON
    json_path = os.path.join(args.outdir, "debug_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print(f" - {csv_path}")
    print(f" - {json_path}")
    print("\nInterpretation:")
    print(" • Mean baseline: train R² ≈ 0; valid R² often slightly < 0.")
    print(" • Linear baseline: should beat mean on train and usually on valid.")
    print(" • NN untrained: near mean baseline.")
    print(" • NN trained: should beat both on train and ideally on valid/test; if not, suspect data/normalization/masking/training setup.")

if __name__ == "__main__":
    main()
