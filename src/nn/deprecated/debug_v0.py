#!/usr/bin/env python
"""
debug.py (Experiment 1 only)

Purpose:
- Build and evaluate four baselines in a self-contained script:
  1) MeanModel       : single constant (fit on TRAIN in physical units)
  2) LinearModel     : pure linear PyTorch model (no activations), fit via least squares in z-scored log1p space
  3) UntrainedNN     : NN architecture (BatchNorm + GELU) with random weights
  4) TrainedNN       : same NN, loads checkpoint nn_exp_1.pth if present

Outputs:
- NetCDF predictions (mm/day) for TRAIN and VALID:
    <resultsdir>/debug/debug_exp_1_<model>_<split>_pr.nc
"""

import os
import json
import time
import numpy as np
import xarray as xr
import torch

# -------------------- Load config --------------------
with open('configs.json','r',encoding='utf-8') as f:
    CONFIGS = json.load(f)

FILEDIR    = CONFIGS['paths']['filedir']
MODELDIR   = CONFIGS['paths']['modeldir']
RESULTSDIR = CONFIGS['paths']['resultsdir']
RUNCONFIGS = CONFIGS['runs']

# Only Experiment 1
EXP_NAME  = 'exp_1'
EXP_CFG   = next((rc for rc in RUNCONFIGS if rc['run_name']==EXP_NAME), RUNCONFIGS[0])
INPUTVARS = EXP_CFG['input_vars']
DESC      = EXP_CFG.get('description', EXP_NAME)

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE     = torch.float32
INF_BATCH = 131072  # inference batch size
OUTDIR    = os.path.join(RESULTSDIR, 'debug')
os.makedirs(OUTDIR, exist_ok=True)

# -------------------- Utilities --------------------
def reshape(da: xr.DataArray) -> np.ndarray:
    """(time,lat,lon[,lev]) -> (N, nfeat)"""
    if 'lev' in da.dims:
        return da.transpose('time','lat','lon','lev').values.reshape(-1, da.lev.size)
    return da.transpose('time','lat','lon').values.reshape(-1, 1)

def load_split(splitname: str, inputvars, target='pr'):
    """
    Returns:
    - X (np.float32, N x D)      : inputs
    - y_z (np.float32, N,)       : z-scored log1p(pr)
    - y_template (xr.DataArray)  : to reshape predictions back to (time,lat,lon)
    """
    assert splitname in ('norm_train','norm_valid')
    path = os.path.join(FILEDIR, f'{splitname}.h5')
    varlist = list(inputvars) + [target]
    ds = xr.open_dataset(path, engine='h5netcdf')[varlist]
    # Match your training windows
    if splitname == 'norm_train':
        ds = ds.sel(time=slice('2011-06-01','2014-08-31'))
    else:
        ds = ds.sel(time=slice('2015-06-01','2015-08-31'))
    Xlist = [reshape(ds[v]) for v in inputvars]
    X = np.concatenate(Xlist, axis=1) if len(Xlist)>1 else Xlist[0]
    y_z = reshape(ds[target]).astype(np.float32).ravel()
    return X.astype(np.float32), y_z, ds[target]

def load_stats():
    with open(os.path.join(FILEDIR, 'stats.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def denorm_to_mm_per_day(y_z: np.ndarray, stats: dict) -> np.ndarray:
    """z-scored log1p(pr) -> mm/day (>=0)"""
    y_log = y_z * float(stats['pr_std']) + float(stats['pr_mean'])
    pr = np.expm1(y_log)
    return np.clip(pr, 0.0, None)

def to_da(y_mm_flat: np.ndarray, template: xr.DataArray, name='predpr') -> xr.DataArray:
    """Flat vector -> xr.DataArray like template (time,lat,lon)"""
    return xr.DataArray(
        y_mm_flat.reshape(template.shape),
        dims=template.dims,
        coords=template.coords,
        name=name,
        attrs={'long_name':'predicted precipitation', 'units':'mm/day'}
    )

def save_nc(da: xr.DataArray, path: str):
    da.to_netcdf(path, engine='h5netcdf')
    with xr.open_dataset(path, engine='h5netcdf'):
        pass

def predict_in_batches(model: torch.nn.Module, X_np: np.ndarray, device=DEVICE, batch=INF_BATCH) -> np.ndarray:
    """Forward (N,D) through PyTorch model -> (N,) (still in z-scored log1p space)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch):
            xb = torch.from_numpy(X_np[i:i+batch]).to(device=device, dtype=DTYPE, non_blocking=True)
            yb = model(xb).squeeze(-1).cpu().numpy()
            outs.append(yb)
    return np.concatenate(outs, axis=0)

# -------------------- Models (self-contained) --------------------
class MeanModel:
    """
    Purpose: Predict a single constant (in mm/day) learned from TRAIN.
    """
    def __init__(self):
        self.const_mm = None

    def fit(self, ytrain_z: np.ndarray, stats: dict):
        ytrain_mm = denorm_to_mm_per_day(ytrain_z, stats)
        self.const_mm = float(np.nanmean(ytrain_mm))

    def predict_mm(self, X_np: np.ndarray) -> np.ndarray:
        assert self.const_mm is not None, "MeanModel not fit yet."
        return np.full(X_np.shape[0], self.const_mm, dtype=np.float32)

class LinearModel(torch.nn.Module):
    """
    Purpose: Pure linear y = XW + b (z-scored log1p space), solved by least squares.
    """
    def __init__(self, inputsize: int, bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(inputsize, 1, bias=bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)

    def fit_closed_form(self, Xtr_np: np.ndarray, ytr_z_np: np.ndarray, device=DEVICE):
        Xtr = torch.from_numpy(Xtr_np).to(device=device, dtype=DTYPE)
        ytr = torch.from_numpy(ytr_z_np).to(device=device, dtype=DTYPE)
        ones = torch.ones((Xtr.shape[0], 1), device=device, dtype=DTYPE)
        X_aug = torch.cat([Xtr, ones], dim=1)  # (N, D+1)
        sol = torch.linalg.lstsq(X_aug, ytr).solution  # (D+1,)
        with torch.no_grad():
            self.linear.weight.copy_(sol[:-1].unsqueeze(0))  # (1,D)
            self.linear.bias.copy_(sol[-1])                  # ()

    def predict_mm(self, X_np: np.ndarray, stats: dict) -> np.ndarray:
        y_z = predict_in_batches(self, X_np)
        return denorm_to_mm_per_day(y_z, stats)

class NNModel(torch.nn.Module):
    """
    Purpose: Same architecture as your nn/model.py (BatchNorm + GELU).
    """
    def __init__(self, inputsize: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,256), torch.nn.BatchNorm1d(256), torch.nn.GELU(),
            torch.nn.Linear(256,128),       torch.nn.BatchNorm1d(128), torch.nn.GELU(),
            torch.nn.Linear(128,64),        torch.nn.BatchNorm1d(64),  torch.nn.GELU(),
            torch.nn.Linear(64,32),         torch.nn.BatchNorm1d(32),  torch.nn.GELU(),
            torch.nn.Linear(32,1)
        )
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)

class UntrainedNN:
    """
    Purpose: NN with random weights (no training). Outputs are in z-scored log1p space, then denormalized.
    """
    def __init__(self, inputsize: int, device=DEVICE):
        self.model = NNModel(inputsize).to(device)
        self.device = device

    def predict_mm(self, X_np: np.ndarray, stats: dict) -> np.ndarray:
        y_z = predict_in_batches(self.model, X_np, device=self.device)
        return denorm_to_mm_per_day(y_z, stats)

class TrainedNN:
    """
    Purpose: NN with trained weights loaded from MODELDIR/nn_exp_1.pth (if present).
    """
    def __init__(self, inputsize: int, ckpt_path: str, device=DEVICE):
        self.model = NNModel(inputsize).to(device)
        self.device = device
        self.ckpt_path = ckpt_path
        self.loaded = False
        if os.path.exists(self.ckpt_path):
            state = torch.load(self.ckpt_path, map_location=device)
            self.model.load_state_dict(state)
            self.loaded = True

    def predict_mm(self, X_np: np.ndarray, stats: dict) -> np.ndarray:
        if not self.loaded:
            raise FileNotFoundError(f"Checkpoint missing: {self.ckpt_path}")
        y_z = predict_in_batches(self.model, X_np, device=self.device)
        return denorm_to_mm_per_day(y_z, stats)

# -------------------- Main --------------------
if __name__ == '__main__':
    t0 = time.time()
    print(f'\n=== Debugging {DESC} ({EXP_NAME}) ===')
    print(f'Device: {DEVICE}')
    print(f'Inputs: {INPUTVARS}\n')

    stats = load_stats()

    # Load splits
    Xtr, ytr_z, ytr_tmpl = load_split('norm_train', INPUTVARS)
    Xva, yva_z, yva_tmpl = load_split('norm_valid', INPUTVARS)

    # 1) MeanModel
    mean_model = MeanModel()
    mean_model.fit(ytr_z, stats)
    yhat_tr_mean = mean_model.predict_mm(Xtr)
    yhat_va_mean = mean_model.predict_mm(Xva)

    # 2) LinearModel (closed-form in z-space)
    lin_model = LinearModel(Xtr.shape[1]).to(DEVICE)
    lin_model.fit_closed_form(Xtr, ytr_z, device=DEVICE)
    yhat_tr_lin = lin_model.predict_mm(Xtr, stats)
    yhat_va_lin = lin_model.predict_mm(Xva, stats)

    # 3) UntrainedNN
    nn_untrained = UntrainedNN(Xtr.shape[1], device=DEVICE)
    yhat_tr_u = nn_untrained.predict_mm(Xtr, stats)
    yhat_va_u = nn_untrained.predict_mm(Xva, stats)

    # 4) TrainedNN (if checkpoint exists)
    ckpt = os.path.join(MODELDIR, f'nn_{EXP_NAME}.pth')
    have_trained = os.path.exists(ckpt)
    if have_trained:
        nn_trained = TrainedNN(Xtr.shape[1], ckpt_path=ckpt, device=DEVICE)
        yhat_tr_t = nn_trained.predict_mm(Xtr, stats)
        yhat_va_t = nn_trained.predict_mm(Xva, stats)
    else:
        print(f'[WARN] Missing trained checkpoint: {ckpt}')

    # Save all predictions as NetCDF (mm/day)
    payloads = [
        ('mean',          yhat_tr_mean, yhat_va_mean),
        ('linear',        yhat_tr_lin,  yhat_va_lin),
        ('nn_untrained',  yhat_tr_u,    yhat_va_u),
    ]
    if have_trained:
        payloads.append(('nn_trained', yhat_tr_t, yhat_va_t))

    for label, ytr_mm, yva_mm in payloads:
        da_tr = to_da(ytr_mm, ytr_tmpl, name='predpr')
        da_va = to_da(yva_mm, yva_tmpl, name='predpr')
        p_tr  = os.path.join(OUTDIR, f'debug_{EXP_NAME}_{label}_train_pr.nc')
        p_va  = os.path.join(OUTDIR, f'debug_{EXP_NAME}_{label}_valid_pr.nc')
        save_nc(da_tr, p_tr)
        save_nc(da_va, p_va)
        print(f'[{label:>12}] saved: {os.path.basename(p_tr)}, {os.path.basename(p_va)}')

    print(f'\nAll done. Output directory: {OUTDIR}')
    print(f'Runtime: {time.time()-t0:.1f}s\n')

