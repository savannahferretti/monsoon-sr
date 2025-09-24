#!/usr/bin/env python

import os
import json
import time
import torch
import logging
import warnings
import numpy as np
import xarray as xr
import wandb

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ---------------------- Configs ----------------------
with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
RESULTSDIR  = CONFIGS['paths']['resultsdir']
INPUTVAR    = 'bl'
TARGETVAR   = 'pr'
DESCRIPTION = 'Experiment 1 (BL-only)'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training knobs
EPOCHS         = 20
BATCHSIZE      = 235_224        # divisible by both sets; change if you want more SGD noise
LEARNINGRATE   = 5e-4           # slightly lower for big-batch stability
WEIGHT_DECAY   = 1e-5
CLIP_MAXNORM   = 1.0
WARMUP_EPOCHS  = 2
PLATEAU_PATIENCE = 2
EARLY_STOP_PATIENCE = 6

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE=='cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CRITERION = torch.nn.MSELoss(reduction='mean')

# ---------------------- Utils -----------------------
def reshape(da):
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1, da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1, 1)
    return arr

def load(splitname, inputvar=INPUTVAR, targetvar=TARGETVAR, filedir=FILEDIR):
    if splitname not in ('norm_train','norm_valid'):
        raise ValueError("Split name must be 'norm_train' or 'norm_valid'")
    filepath = os.path.join(filedir, f'{splitname}.h5')
    ds = xr.open_dataset(filepath, engine='h5netcdf')[[inputvar, targetvar]]
    X = torch.tensor(reshape(ds[inputvar]), dtype=torch.float32)
    y = torch.tensor(reshape(ds[targetvar]), dtype=torch.float32)
    ytemplate = ds[targetvar]
    return X, y, ytemplate

# ---------------------- Baselines -------------------
def fit_mean_model(ytrainflat: np.ndarray) -> float:
    return float(np.mean(ytrainflat))

def predict_mean_model(mu: float, n: int) -> np.ndarray:
    return np.full((n,), mu, dtype=np.float32)

def fit_linear_regression_model(Xtrain: torch.Tensor, ytrain: torch.Tensor, chunksize: int = BATCHSIZE):
    """Streaming univariate OLS in normalized space: y ≈ intercept + slope * x."""
    assert Xtrain.ndim==2 and Xtrain.shape[1]==1, "Expect Xtrain shape (N,1)"
    assert ytrain.shape == Xtrain.shape, "ytrain must match X shape"
    nsamples = Xtrain.shape[0]
    if nsamples == 0:
        return 0.0, 0.0

    # Pass 1: means
    feature_sum = 0.0
    target_sum  = 0.0
    for start in range(0, nsamples, chunksize):
        stop = start + chunksize
        x_chunk = Xtrain[start:stop, 0].detach().cpu().numpy()
        y_chunk = ytrain[start:stop, 0].detach().cpu().numpy()
        feature_sum += x_chunk.sum(dtype=np.float64)
        target_sum  += y_chunk.sum(dtype=np.float64)
    feature_mean = feature_sum / nsamples
    target_mean  = target_sum  / nsamples

    # Pass 2: unnormalized variance & covariance
    sum_sq_feature_dev = 0.0
    sum_feature_target_dev = 0.0
    for start in range(0, nsamples, chunksize):
        stop = start + chunksize
        x_chunk = Xtrain[start:stop, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        y_chunk = ytrain[start:stop, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        x_dev = x_chunk - feature_mean
        y_dev = y_chunk - target_mean
        sum_sq_feature_dev     += np.dot(x_dev, x_dev)
        sum_feature_target_dev += np.dot(x_dev, y_dev)

    if sum_sq_feature_dev <= 0.0:
        slope = 0.0
        intercept = float(target_mean)
        return slope, intercept

    slope     = float(sum_feature_target_dev / sum_sq_feature_dev)
    intercept = float(target_mean - slope * feature_mean)
    return slope, intercept

def predict_linear_regression_model(slope: float, intercept: float, X: torch.Tensor, chunksize: int = BATCHSIZE):
    nsamples = X.shape[0]
    ypred = np.empty((nsamples,), dtype=np.float32)
    for start in range(0, nsamples, chunksize):
        stop = start + chunksize
        x_chunk = X[start:stop, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        ypred[start:stop] = (intercept + slope * x_chunk).astype(np.float32, copy=False)
    return ypred

# ---------------------- NN --------------------------
class NNModel(torch.nn.Module):
    """
    Small, expressive MLP for 1D input:
      1 -> 128 (SiLU) -> 64 (SiLU) -> 32 (SiLU) -> 1
    SiLU is smooth and often works well on physical signals.
    """
    def __init__(self, inputsize: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )
    def forward(self, X):
        return self.layers(X)

def fit_nn_model(
    Xtrain, ytrain, Xvalid=None, yvalid=None,
    epochs=EPOCHS, batchsize=BATCHSIZE, learningrate=LEARNINGRATE,
    weight_decay=WEIGHT_DECAY, clip_maxnorm=CLIP_MAXNORM,
    warmup_epochs=WARMUP_EPOCHS, plateau_patience=PLATEAU_PATIENCE,
    early_stop_patience=EARLY_STOP_PATIENCE, wandb_prefix="nn_trained"
):
    model = NNModel(Xtrain.shape[1]).to(DEVICE)

    trainset = torch.utils.data.TensorDataset(Xtrain, ytrain)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    validloader = None
    if Xvalid is not None and yvalid is not None:
        validset = torch.utils.data.TensorDataset(Xvalid, yvalid)
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=batchsize, shuffle=False,
            num_workers=8, pin_memory=True, persistent_workers=True
        )

    opt = torch.optim.AdamW(model.parameters(), lr=learningrate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    best_val = float('inf')
    best_state = None
    no_improve = 0
    base_lr = learningrate

    for epoch in range(1, epochs+1):
        # Warmup
        if epoch <= warmup_epochs and warmup_epochs > 0:
            for pg in opt.param_groups:
                pg['lr'] = base_lr * epoch / warmup_epochs

        # Train epoch
        model.train()
        train_running, train_count = 0.0, 0
        for xb, yb in trainloader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).squeeze(-1)
            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_maxnorm)
            opt.step()
            train_running += loss.item() * xb.size(0)
            train_count  += xb.size(0)
        train_loss = train_running / max(train_count, 1)

        # Validate
        if validloader is not None:
            model.eval()
            valid_running, valid_count = 0.0, 0
            with torch.no_grad():
                for (xb, yb) in validloader:
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True).squeeze(-1)
                    pred = model(xb).squeeze(-1)
                    loss = loss_fn(pred, yb)
                    valid_running += loss.item() * xb.size(0)
                    valid_count  += xb.size(0)
            valid_loss = valid_running / max(valid_count, 1)
        else:
            valid_loss = float('nan')

        # --- Only log LOSSES (no extra metrics) ---
        wandb.log({
            f'{wandb_prefix}/epoch': epoch,
            f'{wandb_prefix}/train_MSE': train_loss,
            f'{wandb_prefix}/valid_MSE': valid_loss,
            f'{wandb_prefix}/lr': opt.param_groups[0]['lr'],
        })

        # Track best val + simple LR plateau + early stop
        improved = (valid_loss < best_val) if not np.isnan(valid_loss) else False
        if improved:
            best_val = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= plateau_patience:
                for pg in opt.param_groups:
                    pg['lr'] = max(pg['lr'] * 0.5, base_lr * 1e-2)
                no_improve = 0  # reset LR patience

        if not np.isnan(valid_loss) and (no_improve >= early_stop_patience):
            logger.info("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_nn_model(model, X, batchsize=BATCHSIZE):
    evaldataset = torch.utils.data.TensorDataset(X)
    evalloader  = torch.utils.data.DataLoader(
        evaldataset, batch_size=batchsize, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    outs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in evalloader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = model(xb).squeeze(-1).detach().cpu().numpy()
            outs.append(yb)
    return np.concatenate(outs, axis=0)

# ---------------------- Main ------------------------
if __name__=='__main__':
    try:
        start = time.time()
        run = wandb.init(
            project='Precipitation NNs Debug',
            name='debug_exp1_BL_only_losses_only',
            config={
                'description': DESCRIPTION,
                'epochs': EPOCHS,
                'batch_size': BATCHSIZE,
                'learning_rate': LEARNINGRATE,
                'weight_decay': WEIGHT_DECAY,
                'clip_maxnorm': CLIP_MAXNORM,
                'warmup_epochs': WARMUP_EPOCHS,
                'plateau_patience': PLATEAU_PATIENCE,
                'early_stop_patience': EARLY_STOP_PATIENCE,
                'device': DEVICE,
            }
        )

        logger.info(f'Device: {DEVICE}')
        logger.info('Loading training/validation splits for Experiment 1...')
        Xtrainnorm, ytrainnorm, ytraintemplate = load('norm_train')
        Xvalidnorm, yvalidnorm, yvalidtemplate = load('norm_valid')

        # Flat normalized truth vectors (for baselines’ losses)
        ytrainnormflat = ytrainnorm.squeeze(-1).numpy().ravel().astype(np.float32)
        yvalidnormflat = yvalidnorm.squeeze(-1).numpy().ravel().astype(np.float32)

        # ---------- Step 1: Mean model (losses only) ----------
        logger.info('Mean model...')
        mu_norm = fit_mean_model(ytrainnormflat)
        ytrainnorm_pred1 = predict_mean_model(mu_norm, n=ytrainnormflat.size)
        yvalidnorm_pred1 = predict_mean_model(mu_norm, n=yvalidnormflat.size)

        # Compute losses via torch to be consistent
        loss_train_mean = CRITERION(
            torch.tensor(ytrainnorm_pred1), torch.tensor(ytrainnormflat)
        ).item()
        loss_valid_mean = CRITERION(
            torch.tensor(yvalidnorm_pred1), torch.tensor(yvalidnormflat)
        ).item()
        wandb.log({'mean/train_MSE': loss_train_mean, 'mean/valid_MSE': loss_valid_mean})

        # ---------- Step 2: Linear regression (losses only) ----------
        logger.info('Linear regression (streaming OLS)...')
        slope, intercept = fit_linear_regression_model(Xtrainnorm, ytrainnorm)
        ytrainnorm_pred2 = predict_linear_regression_model(slope, intercept, Xtrainnorm)
        yvalidnorm_pred2 = predict_linear_regression_model(slope, intercept, Xvalidnorm)

        loss_train_lin = CRITERION(
            torch.tensor(ytrainnorm_pred2), torch.tensor(ytrainnormflat)
        ).item()
        loss_valid_lin = CRITERION(
            torch.tensor(yvalidnorm_pred2), torch.tensor(yvalidnormflat)
        ).item()
        wandb.log({'linear/train_MSE': loss_train_lin, 'linear/valid_MSE': loss_valid_lin})

        # ---------- Step 3: Untrained NN (losses only) ----------
        logger.info('Untrained NN...')
        nn_untrained = NNModel(inputsize=Xtrainnorm.shape[1]).to(DEVICE)
        ytrainnorm_pred3 = predict_nn_model(nn_untrained, Xtrainnorm)
        yvalidnorm_pred3 = predict_nn_model(nn_untrained, Xvalidnorm)

        loss_train_nnu = CRITERION(
            torch.tensor(ytrainnorm_pred3), torch.tensor(ytrainnormflat)
        ).item()
        loss_valid_nnu = CRITERION(
            torch.tensor(yvalidnorm_pred3), torch.tensor(yvalidnormflat)
        ).item()
        wandb.log({'nn_untrained/train_MSE': loss_train_nnu, 'nn_untrained/valid_MSE': loss_valid_nnu})

        # ---------- Step 4: Trained NN (per-epoch losses only) ----------
        logger.info('Trained NN...')
        nn_trained = fit_nn_model(
            Xtrainnorm, ytrainnorm,
            Xvalid=Xvalidnorm, yvalid=yvalidnorm,
            epochs=EPOCHS, batchsize=BATCHSIZE, learningrate=LEARNINGRATE,
            wandb_prefix='nn_trained'
        )

        # Final losses after restoring best state (optional, but useful)
        with torch.no_grad():
            # train loss
            trainset = torch.utils.data.TensorDataset(Xtrainnorm, ytrainnorm)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=BATCHSIZE, shuffle=False,
                num_workers=8, pin_memory=True, persistent_workers=True
            )
            nn_trained.eval()
            tr_run, tr_cnt = 0.0, 0
            for xb, yb in trainloader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).squeeze(-1)
                pred = nn_trained(xb).squeeze(-1)
                loss = CRITERION(pred, yb)
                tr_run += loss.item() * xb.size(0); tr_cnt += xb.size(0)
            final_train_loss = tr_run / max(tr_cnt, 1)

            # valid loss
            validset = torch.utils.data.TensorDataset(Xvalidnorm, yvalidnorm)
            validloader = torch.utils.data.DataLoader(
                validset, batch_size=BATCHSIZE, shuffle=False,
                num_workers=8, pin_memory=True, persistent_workers=True
            )
            va_run, va_cnt = 0.0, 0
            for xb, yb in validloader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).squeeze(-1)
                pred = nn_trained(xb).squeeze(-1)
                loss = CRITERION(pred, yb)
                va_run += loss.item() * xb.size(0); va_cnt += xb.size(0)
            final_valid_loss = va_run / max(va_cnt, 1)

        wandb.log({'nn_trained/final_train_MSE': final_train_loss,
                   'nn_trained/final_valid_MSE': final_valid_loss})

        # ---------- Save VALID predictions (physical) ----------
        # (Keeping your single-file output for downstream analysis)
        logger.info('Saving validation set predictions to a single NetCDF...')
        def _denorm_vec(v):
            # read stats once
            with open(os.path.join(FILEDIR,'stats.json'),'r',encoding='utf-8') as f:
                stats = json.load(f)
            mu, sd = float(stats['pr_mean']), float(stats['pr_std'])
            return np.expm1(v*sd + mu)

        yvalid_mean    = _denorm_vec(predict_mean_model(mu_norm, n=yvalidnormflat.size))
        yvalid_linear  = _denorm_vec(predict_linear_regression_model(slope, intercept, Xvalidnorm))
        yvalid_nnu     = _denorm_vec(predict_nn_model(nn_untrained, Xvalidnorm))
        yvalid_nn      = _denorm_vec(predict_nn_model(nn_trained,  Xvalidnorm))

        os.makedirs(RESULTSDIR, exist_ok=True)
        outpath = os.path.join(RESULTSDIR, 'debug_exp1_norm_valid_pr.nc')
        da1 = xr.DataArray(yvalid_mean.reshape(yvalidtemplate.shape),  dims=yvalidtemplate.dims,  coords=yvalidtemplate.coords,  name='predpr_mean')
        da2 = xr.DataArray(yvalid_linear.reshape(yvalidtemplate.shape),dims=yvalidtemplate.dims, coords=yvalidtemplate.coords, name='predpr_linear')
        da3 = xr.DataArray(yvalid_nnu.reshape(yvalidtemplate.shape),   dims=yvalidtemplate.dims, coords=yvalidtemplate.coords,   name='predpr_nn_untrained')
        da4 = xr.DataArray(yvalid_nn.reshape(yvalidtemplate.shape),    dims=yvalidtemplate.dims, coords=yvalidtemplate.coords,    name='predpr_nn_trained')
        dsout = xr.Dataset({da1.name: da1, da2.name: da2, da3.name: da3, da4.name: da4})
        dsout.to_netcdf(outpath, engine='h5netcdf')
        with xr.open_dataset(outpath, engine='h5netcdf') as _:
            pass
        logger.info(f'Wrote {outpath}')

        wandb.finish()
        logger.info(f'Done in {time.time()-start:.1f}s!')

    except Exception as e:
        logger.exception(f'An unexpected error occurred: {e}')
        try:
            wandb.alert(title="debug_exp1 failed", text=str(e))
            wandb.finish()
        except Exception:
            pass
        raise