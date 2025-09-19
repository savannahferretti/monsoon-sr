#!/usr/bin/env python

import os
import json
import time
import torch
import numpy as np
import xarray as xr
from torch.utils.data import TensorDataset, DataLoader

# -------------------- Config --------------------
with open('configs.json','r',encoding='utf-8') as f:
    CONFIGS = json.load(f)

FILEDIR    = CONFIGS['paths']['filedir']
MODELDIR   = CONFIGS['paths']['modeldir']
RESULTSDIR = CONFIGS['paths']['resultsdir']
RUNCONFIGS = CONFIGS['runs']

EXP_NAME    = 'exp_1'
EXP_CONFIG  = next((rc for rc in RUNCONFIGS if rc['run_name']==EXP_NAME), RUNCONFIGS[0])
INPUTVARS   = EXP_CONFIG['input_vars']
DESCRIPTION = EXP_CONFIG.get('description', EXP_NAME)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32

# Training hyperparams
TRAINBATCH     = 92
EPOCHS         = 8
LR             = 1e-3
WEIGHTDECAY    = 1e-4
SCHEDPATIENCE  = 2
STOPPATIENCE   = 6
CLIPMAXNORM    = 1.0
POSWEIGHT      = 8.0  # rain-aware weighting factor

OUTDIR = os.path.join(RESULTSDIR,'debug')
os.makedirs(OUTDIR,exist_ok=True)

# -------------------- Utilities --------------------
def reshape(da: xr.DataArray) -> np.ndarray:
    """(time,lat,lon[,lev]) -> (N, nfeat)"""
    if 'lev' in da.dims:
        return da.transpose('time','lat','lon','lev').values.reshape(-1, da.lev.size)
    return da.transpose('time','lat','lon').values.reshape(-1, 1)

def load_split(splitname: str, inputvars, target='pr'):
    assert splitname in ('norm_train','norm_valid')
    path = os.path.join(FILEDIR, f'{splitname}.h5')
    varlist = list(inputvars) + [target]
    ds = xr.open_dataset(path, engine='h5netcdf')[varlist]
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

def predict_full(model: torch.nn.Module, X_np: np.ndarray) -> np.ndarray:
    """
    Non-batched inference (as requested).
    NOTE: This moves the entire array to GPU/CPU at once; ensure it fits in memory.
    """
    model.eval()
    with torch.inference_mode():
        xb = torch.from_numpy(X_np).to(device=next(model.parameters()).device, dtype=DTYPE)
        yb = model(xb).squeeze(-1).cpu().numpy()
    return yb

# -------------------- Models --------------------
class MeanModel:
    """Predict a single constant (in mm/day) learned from TRAIN."""
    def __init__(self):
        self.const_mm = None
    def fit(self, ytrain_z: np.ndarray, stats: dict):
        ytrain_mm = denorm_to_mm_per_day(ytrain_z, stats)
        self.const_mm = float(np.nanmean(ytrain_mm))
    def predict_mm(self, X_np: np.ndarray) -> np.ndarray:
        assert self.const_mm is not None, "MeanModel not fit yet."
        return np.full(X_np.shape[0], self.const_mm, dtype=np.float32)

class LinearModel(torch.nn.Module):
    """Pure linear y = XW + b (z-scored log1p space), solved by least squares."""
    def __init__(self, inputsize: int, bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(inputsize, 1, bias=bias)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)
    def fit_closed_form(self, Xtr_np: np.ndarray, ytr_z_np: np.ndarray):
        X_aug = np.concatenate([Xtr_np, np.ones((Xtr_np.shape[0], 1), dtype=np.float64)], axis=1).astype(np.float64, copy=False)
        y_tr  = ytr_z_np.astype(np.float64, copy=False)
        beta, *_ = np.linalg.lstsq(X_aug, y_tr, rcond=None)  # (D+1,)
        w = torch.from_numpy(beta[:-1].astype(np.float32))
        b = torch.tensor(beta[-1], dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.copy_(w.unsqueeze(0))  # (1,D)
            self.linear.bias.copy_(b)
    def predict_mm(self, X_np: np.ndarray, stats: dict) -> np.ndarray:
        y_z = predict_full(self.to(DEVICE), X_np)
        return denorm_to_mm_per_day(y_z, stats)

class NNModelNoBN(torch.nn.Module):
    """BN-free MLP with GELU activations."""
    def __init__(self, inputsize: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,256), torch.nn.GELU(),
            torch.nn.Linear(256,128),       torch.nn.GELU(),
            torch.nn.Linear(128,64),        torch.nn.GELU(),
            torch.nn.Linear(64,32),         torch.nn.GELU(),
            torch.nn.Linear(32,1)
        )
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)

class UntrainedNN:
    """NN with random weights (no training). Outputs are in z-scored log1p, then denormalized."""
    def __init__(self, inputsize: int, device: torch.device):
        self.model  = NNModelNoBN(inputsize).to(device)
    def predict_mm(self, X_np: np.ndarray, stats: dict) -> np.ndarray:
        y_z = predict_full(self.model, X_np)
        return denorm_to_mm_per_day(y_z, stats)

# -------------------- Training (rain-aware) --------------------
def train_nn_rain_aware(inputsize: int,
                        Xtr_np: np.ndarray, ytr_z_np: np.ndarray,
                        Xva_np: np.ndarray, yva_z_np: np.ndarray,
                        stats: dict) -> NNModelNoBN:
    """
    Train a BN-free MLP using a rain-aware MSE:
      loss = mean( w * (y_hat - y)^2 ), where w = POSWEIGHT if (rain), else 1
    We detect 'rain' using the z-space threshold z_thresh = -mean/std.
    """
    z_thresh = -float(stats['pr_mean']) / float(stats['pr_std'])

    model = NNModelNoBN(inputsize).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHTDECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=SCHEDPATIENCE, threshold=1e-4, cooldown=1, verbose=False
    )
    crit  = torch.nn.MSELoss(reduction='none')

    Xtr = torch.from_numpy(Xtr_np)
    ytr = torch.from_numpy(ytr_z_np)
    Xva = torch.from_numpy(Xva_np)
    yva = torch.from_numpy(yva_z_np)

    trainloader = DataLoader(TensorDataset(Xtr,ytr), batch_size=TRAINBATCH, shuffle=True,
                             num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    validloader = DataLoader(TensorDataset(Xva,yva), batch_size=TRAINBATCH, shuffle=False,
                             num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    best = float('inf'); best_state = None; no_improve = 0
    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        runloss = 0.0; nobs = 0
        for xb, yb in trainloader:
            if DEVICE.type == 'cuda':
                xb = xb.pin_memory().to(DEVICE, DTYPE, non_blocking=True)
                yb = yb.pin_memory().to(DEVICE, DTYPE, non_blocking=True)
            else:
                xb = xb.to(DEVICE, DTYPE)
                yb = yb.to(DEVICE, DTYPE)

            opt.zero_grad(set_to_none=True)
            yp = model(xb).squeeze(-1)
            per = crit(yp, yb)
            w = torch.where(yb > z_thresh, float(POSWEIGHT), 1.0)
            loss = (w * per).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPMAXNORM)
            opt.step()
            runloss += float(loss.item()) * xb.size(0)
            nobs    += xb.size(0)
        trainloss = runloss / max(1, nobs)

        # ---- valid ----
        model.eval()
        valloss = 0.0; nobs = 0
        with torch.inference_mode():
            for xb, yb in validloader:
                if DEVICE.type == 'cuda':
                    xb = xb.pin_memory().to(DEVICE, DTYPE, non_blocking=True)
                    yb = yb.pin_memory().to(DEVICE, DTYPE, non_blocking=True)
                else:
                    xb = xb.to(DEVICE, DTYPE)
                    yb = yb.to(DEVICE, DTYPE)
                yp = model(xb).squeeze(-1)
                per = crit(yp, yb)
                w = torch.where(yb > z_thresh, float(POSWEIGHT), 1.0)
                loss = (w * per).mean()
                valloss += float(loss.item()) * xb.size(0)
                nobs    += xb.size(0)
        valloss /= max(1, nobs)
        sched.step(valloss)

        improved = valloss < best - 1e-7
        if improved:
            best = valloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= STOPPATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -------------------- Main --------------------
if __name__ == '__main__':
    t0 = time.time()
    print(f'\n=== Debugging {DESCRIPTION} ({EXP_NAME}) ===')
    print(f'Device: {DEVICE.type}')
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
    lin_model.fit_closed_form(Xtr, ytr_z)
    yhat_tr_lin = denorm_to_mm_per_day(predict_full(lin_model, Xtr), stats)
    yhat_va_lin = denorm_to_mm_per_day(predict_full(lin_model, Xva), stats)

    # 3) NN (Untrained, no BN)
    nn_untrained = UntrainedNN(Xtr.shape[1], device=DEVICE)
    yhat_tr_u = nn_untrained.predict_mm(Xtr, stats)
    yhat_va_u = nn_untrained.predict_mm(Xva, stats)

    # 4) NN (Trained here with rain-aware loss, no BN)
    nn_trained = train_nn_rain_aware(Xtr.shape[1], Xtr, ytr_z, Xva, yva_z, stats)
    yhat_tr_t = denorm_to_mm_per_day(predict_full(nn_trained, Xtr), stats)
    yhat_va_t = denorm_to_mm_per_day(predict_full(nn_trained, Xva), stats)

    # Save all predictions as NetCDF (mm/day)
    payloads = [
        ('mean',          yhat_tr_mean, yhat_va_mean),
        ('linear',        yhat_tr_lin,  yhat_va_lin),
        ('nn_untrained',  yhat_tr_u,    yhat_va_u),
        ('nn_trained',    yhat_tr_t,    yhat_va_t),
    ]

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

