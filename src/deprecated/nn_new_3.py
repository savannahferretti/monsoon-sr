#!/usr/bin/env python

import os
import time
import h5py
import torch
import wandb
import pickle
import logging
import warnings
import numpy as np
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/nn'
CONFIGS  = [
    {'name':'exp_1','inputvars':['bl'],'description':'Experiment 1'},
    # {'name':'exp_2','inputvars':['cape','subsat'],'description':'Experiment 2'},
    # {'name':'exp_3','inputvars':['capeprofile'],'description':'Experiment 3'},
    # {'name':'exp_4','inputvars':['subsatprofile'],'description':'Experiment 4'},
    # {'name':'exp_5','inputvars':['capeprofile','subsatprofile'],'description':'Experiment 5'},
    # {'name':'exp_6','inputvars':['t','q'],'description':'Experiment 6'}
]

NORMTARGET  = True
LOG1PTARGET = True
        
class NNMODEL:
    
    def __init__(self,inputsize,batchsize=64000,epochs=30,criterion=torch.nn.MSELoss(),learningrate=0.001,patience=3):
        '''
        Purpose: Initialize a NN model for precipitation prediction.
        '''
        self.inputsize    = inputsize
        self.batchsize    = batchsize
        self.epochs       = epochs
        self.criterion    = criterion
        self.learningrate = learningrate
        self.patience     = patience
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = torch.nn.Sequential(
            torch.nn.Linear(self.inputsize,256),torch.nn.BatchNorm1d(256),torch.nn.GELU(),
            torch.nn.Linear(256,128),torch.nn.BatchNorm1d(128),torch.nn.GELU(),
            torch.nn.Linear(128,64),torch.nn.BatchNorm1d(64),torch.nn.GELU(),
            torch.nn.Linear(64,32),torch.nn.BatchNorm1d(32),torch.nn.GELU(),
            torch.nn.Linear(32,1)).to(self.device)
        self.bestloss     = float('inf')
        self.bestepoch    = 0
        self.beststate    = None

        self.history = {
            'epoch':[],
            'train_loss':[],
            'valid_loss':[],
            'eval_train_mse':[],
            'eval_valid_mse':[]}

    def _eval_mse_evalmode(self, loader):
        """Criterion MSE with model in eval() mode."""
        self.model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(Xb).squeeze()
                loss = self.criterion(pred, yb.squeeze())
                total += loss.item() * Xb.size(0)
                n += Xb.size(0)
        return total / max(n, 1)

    def _save_checkpoint(self, epoch, optimizer, exp_name, checkpoint_root=MODELDIR):
        """Optional: keep per-epoch checkpoints (unchanged)."""
        ckpt_dir = os.path.join(checkpoint_root, 'checkpoints', exp_name if exp_name else 'default')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)
        return path

    def fit(self,Xtrain,Xvalid,ytrain,yvalid,*,exp_name=None,normparams=None,save_all_epochs=True):
        '''
        Train a NN model and log per-epoch diagnostics.
        NOTE: normparams is unused now (kept only for call-site compatibility).
        '''
        trainloader = DataLoader(TensorDataset(Xtrain,ytrain),batch_size=self.batchsize,shuffle=True,num_workers=8,pin_memory=True)
        validloader = DataLoader(TensorDataset(Xvalid,yvalid),batch_size=self.batchsize,shuffle=False,num_workers=8,pin_memory=True)
        optimizer   = torch.optim.Adam(self.model.parameters(),lr=self.learningrate)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=self.patience)
        counter     = 0
        starttime   = time.time()

        for epoch in range(1,self.epochs+1):
            
            self.model.train()
            trainloss = 0.0
            for Xbatch,ybatch in trainloader:
                Xbatch = Xbatch.to(self.device)
                ybatch = ybatch.to(self.device)
                optimizer.zero_grad()
                loss   = self.criterion(self.model(Xbatch).squeeze(),ybatch.squeeze())
                loss.backward()
                optimizer.step()
                trainloss += loss.item()*Xbatch.size(0)
            trainloss /= len(trainloader.dataset)
            if not np.isfinite(trainloss):
                logger.warning('   Training loss is non-finite. Stopping...')
                break


            self.model.eval()
            validloss = 0.0
            with torch.no_grad():
                for Xbatch,ybatch in validloader:
                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)
                    loss   = self.criterion(self.model(Xbatch).squeeze(),ybatch.squeeze())
                    validloss += loss.item()*Xbatch.size(0)
            validloss /= len(validloader.dataset)
            if not np.isfinite(validloss):
                logger.warning('   Validation loss is non-finite. Stopping...')
                break

            # -------- separate evaluation passes (same metric, same space) ----------
            eval_train_evalmode = self._eval_mse_evalmode(trainloader)         # model.eval()
            eval_valid_evalmode = self._eval_mse_evalmode(validloader)         # model.eval()
            eval_train_trainmode = self._eval_mse_trainmode_nograd(trainloader)  # model.train(), no_grad

            # deltas for BN diagnosis
            delta_train_eval     = abs(trainloss - eval_train_evalmode)
            delta_train_trainmod = abs(trainloss - eval_train_trainmode)

            # -------- record history ----------
            self.history['epoch'].append(epoch)
            self.history['train_loss_inloop'].append(trainloss)
            self.history['valid_loss_inloop'].append(validloss)
            self.history['eval_train_mse_evalmode'].append(eval_train_evalmode)
            self.history['eval_valid_mse_evalmode'].append(eval_valid_evalmode)
            self.history['eval_train_mse_trainmode'].append(eval_train_trainmode)
            self.history['delta_train_eval'].append(delta_train_eval)
            self.history['delta_train_trainmode'].append(delta_train_trainmod)

            # -------- LR scheduler / early stopping ----------
            scheduler.step(validloss)

            # -------- logging ----------
            try:
                wandb.log({
                    'Epoch':epoch,
                    'Criterion/Train Loss (in-loop)':trainloss,
                    'Criterion/Valid Loss (in-loop)':validloss,
                    'Criterion/Eval Train MSE (eval)':eval_train_evalmode,
                    'Criterion/Eval Train MSE (train)':eval_train_trainmode,
                    'Criterion/Eval Valid MSE (eval)':eval_valid_evalmode,
                    'Criterion/Δ train (inloop - eval)':delta_train_eval,
                    'Criterion/Δ train (inloop - train)':delta_train_trainmod,
                    'Learning Rate':optimizer.param_groups[0]['lr']})
            except Exception:
                pass

            # concise console line + BN hint
            bn_hint = ""
            # if in-loop ≈ trainmode (tiny delta) but evalmode differs more -> BN likely
            if delta_train_trainmod < 1e-4 and delta_train_eval > 5e-4:
                bn_hint = " | BN likely driver"
            logger.info(
                f'   Ep {epoch:03d} | train {trainloss:.6f} | valid {validloss:.6f} | '
                f'eval-tr(eval) {eval_train_evalmode:.6f} | eval-tr(train) {eval_train_trainmode:.6f} | '
                f'dΔ eval {delta_train_eval:.2e} | dΔ train {delta_train_trainmod:.2e}{bn_hint}'
            )

            # -------- checkpoint every epoch (optional) ----------
            if save_all_epochs:
                self._save_checkpoint(epoch, optimizer, exp_name)

            # -------- early stopping tracking ----------
            if validloss < self.bestloss:
                counter = 0
                self.bestloss  = validloss
                self.bestepoch = epoch
                self.beststate = self.model.state_dict().copy()
            else:
                counter += 1
                if counter > self.patience:
                    logger.info(f'   Early stopping at epoch {epoch}!')
                    break

        # restore best
        if self.beststate is not None:
            self.model.load_state_dict(self.beststate)
        trainingtime = time.time()-starttime
        try:
            wandb.run.summary.update({
                'Best Model at Epoch':self.bestepoch,
                'Best Validation Loss':self.bestloss,
                'Total Training Epochs':epoch,
                'Training Duration (s)':trainingtime,
                'Stopped Early':counter>self.patience})
        except Exception:
            pass
    
    def predict(self,X):
        self.model.eval()
        with torch.no_grad():
            X     = X.to(self.device)
            ypred = self.model(X).squeeze()
        return ypred.cpu().numpy()
    
    def save(self,name,modeldir=MODELDIR):
        filename = f'{name}_best_normtarget_NEW_DEBUG.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(self.model.state_dict(),filepath)
        logger.info(f'   Model saved to {filepath}')

        
def load(inputvars, filename, filedir=FILEDIR):
    def _stack_with_masks(f, split, inputvars):
        blocks = []
        for inputvar in inputvars:
            X = f[f'{inputvar}_{split}'][:]
            if inputvar in ['t','q','capeprofile','subsatprofile']:
                mask = np.isfinite(X).astype(np.float32)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                X = X * mask
                blocks.append(np.concatenate([X, mask], axis=1))
            else:
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                blocks.append(X)
        return torch.tensor(np.concatenate(blocks, axis=1), dtype=torch.float32)

    filepath = os.path.join(filedir, filename)
    with h5py.File(filepath, 'r') as f:
        Xtrain = _stack_with_masks(f, 'train', inputvars)
        Xvalid = _stack_with_masks(f, 'valid', inputvars)
        Xtest  = _stack_with_masks(f, 'test',  inputvars)

        # keep as numpy until normalization is done
        ytrain = f['pr_train'][:].astype(np.float32).squeeze()
        yvalid = f['pr_valid'][:].astype(np.float32).squeeze()
        ytest  = f['pr_test'][:].astype(np.float32).squeeze()

    # define normparams regardless
    normparams = None
    if NORMTARGET:
        if LOG1PTARGET:
            ytrain = np.log1p(ytrain)
            yvalid = np.log1p(yvalid)
            ytest  = np.log1p(ytest)
        ymean = ytrain.mean()
        ystd  = ytrain.std() + 1e-8
        ytrain = (ytrain - ymean) / ystd
        yvalid = (yvalid - ymean) / ystd
        ytest  = (ytest  - ymean) / ystd
        normparams = {'mean': ymean, 'std': ystd, 'log1p': LOG1PTARGET}

    # convert to tensors once
    ytrain = torch.tensor(ytrain, dtype=torch.float32)
    yvalid = torch.tensor(yvalid, dtype=torch.float32)
    ytest  = torch.tensor(ytest,  dtype=torch.float32)

    return Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest, normparams

def invert_normalization(y_norm, normparams):
    if normparams is None:
        return y_norm
    y = y_norm * normparams['std'] + normparams['mean']
    if normparams.get('log1p', False):
        y = np.expm1(y)
    return y
        
def process(filename,configs=CONFIGS):
    '''
    Purpose: Train and evaluate NN models with multiple input variable configurations.
    Args:
    - filename (str): name of the HDF5 file
    - configs (list): model configurations specifying input variables and descriptions (defaults to CONFIGS)
    Returns:
    - dict: dictionary containing NN model results
    '''
    results = {}
    for config in configs:
        name        = config['name']
        inputvars   = config['inputvars']
        description = config['description']
        logger.info(f'   Running {description}')
        Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams = load(inputvars,filename)
        model = NNMODEL(Xtrain.shape[1])
        wandb.init(
            project='Debug NNs',
            name=description,
            config={
                'Input Variables':inputvars,
                'Model Parameter Count':sum(p.numel() for p in model.model.parameters()),
                'Training Batch Size':model.batchsize,
                'Maximum Training Epochs':model.epochs,
                'Network Architecture':'256→128→64→32→1 (GELU + BatchNorm)',
                'Optimizer':'Adam',
                'Loss Function':'Mean Squared Error',
                'Initial Learning Rate':model.learningrate,
                'Learning Rate Scheduler':'ReduceLROnPlateau (factor=0.5)',
                'Early Stopping Patience':model.patience})
        
        model.fit(Xtrain, Xvalid, ytrain, yvalid, exp_name=name, normparams=normparams, save_all_epochs=True)
        
        model.save(name)
        results[name] = {
            'description':description,
            'n_params':sum(p.numel() for p in model.model.parameters()),
            'y_pred':invert_normalization(model.predict(Xtest),normparams)}
        wandb.finish()
        del model,Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
    
if __name__=='__main__':
    try:
        logger.info('Training NN models...')
        results = process('ml_data_subset.h5')
        logger.info('Saving results...')
        with open(f'{SAVEDIR}/nn_normtarget_NEW_subset_results.pkl', 'wb') as f:
            pickle.dump(results,f)
        del results
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')