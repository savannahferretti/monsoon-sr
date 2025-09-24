#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import xarray as xr

SAVEDIR        = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/splits'
NTIMETRAIN     = 7000
NTIMEVALID     = 3000
LONMIN,LONMAX  = 60.0,90.0
LATMIN,LATMAX  = 5.0,25.0
STARTTRAIN     = '2000-01-01'
STARTVALID     = (pd.to_datetime(STARTTRAIN)+pd.Timedelta(days=NTIMETRAIN)).strftime('%Y-%m-%d')
SEEDTRAIN      = 42
SEEDVALID      = 43
NOISESTD       = 0.05

def make_split(ntime,lats,lons,seed,startdate):
    rng  = np.random.default_rng(seed)
    time = pd.date_range(startdate,periods=ntime,freq='D')
    lat  = np.asarray(lats,dtype=np.float32)
    lon  = np.asarray(lons,dtype=np.float32)
    tt   = np.arange(ntime,dtype=np.float32)[:,None,None]
    latg = lat[None,:,None]
    long = lon[None,None,:]
    noise = rng.normal(0.0,NOISESTD,size=(ntime,len(lat),len(lon))).astype(np.float32)
    x = (0.8*np.sin(2*np.pi*tt/30.0)+
         0.1*(latg/max(1e-6,np.max(np.abs(lat))))+
         0.05*(long/max(1e-6,np.max(np.abs(lon))))+noise).astype(np.float32)
    y = (3.0*x+1.5).astype(np.float32)
    ds = xr.Dataset(
        data_vars={
            'x':(('time','lat','lon'),x,{'long_name':'x','units':'arb'}),
            'y':(('time','lat','lon'),y,{'long_name':'y = 3*x + 1.5','units':'arb'})},
        coords={'time':time,'lat':lat,'lon':lon})
    return ds

if __name__=='__main__':
    os.makedirs(SAVEDIR,exist_ok=True)
    lats = np.arange(LATMIN,LATMAX + 0.25,0.25,dtype=np.float32)
    lons = np.arange(LONMIN,LONMAX + 0.25,0.25,dtype=np.float32)
    trainds = make_split(NTIMETRAIN,lats,lons,SEEDTRAIN,STARTTRAIN)
    validds = make_split(NTIMEVALID,lats,lons,SEEDVALID,STARTVALID)
    trainpath = f'{SAVEDIR}/debug_train.h5'
    trainds.to_netcdf(trainpath, engine='h5netcdf')
    print(f'Wrote: {trainpath}')
    validpath = f'{SAVEDIR}/debug_valid.h5'
    validds.to_netcdf(validpath, engine='h5netcdf')
    print(f'Wrote: {validpath}')