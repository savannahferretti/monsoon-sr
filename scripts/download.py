#!/usr/bin/env python

import os
import re
import sys
import fsspec
import logging
import warnings
import numpy as np
import xarray as xr
import planetary_computer
from datetime import datetime
import pystac_client as pystac

logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

AUTHOR    = 'Savannah L. Ferretti'
EMAIL     = 'savannah.ferretti@uci.edu'
SAVEDIR   = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/raw'
YEARS     = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
MONTHS    = [6,7,8]
LATRANGE  = (5.,25.) 
LONRANGE  = (60.,90.)
LEVRANGE  = (500.,1000.)

def retrieve_era5():
    '''
    Purpose: Retrieve the ERA5 (ARCO) Zarr store from Google Cloud and return it as an xr.Dataset.
    Returns:
    - xr.Dataset: ERA5 Dataset on success, the program exists if access fails
    '''
    try:
        store = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
        ds    = xr.open_zarr(store,decode_times=True)  
        logger.info('   Successfully retrieved ERA5')
        return ds
    except Exception:
        logger.exception('   Failed to retrieve ERA5')
        sys.exit(1)

def retrieve_imerg():
    '''
    Purpose: Retrieve the GPM IMERG V06 Zarr store from Microsoft Planetary Computer and return it as an xr.Dataset.
    Returns: 
    - xr.Dataset: IMERG V06 Dataset on success, the program exists if access fails
    '''
    try:
        store   = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        catalog = pystac.Client.open(store, modifier=planetary_computer.sign_inplace)
        assets  = catalog.get_collection('gpm-imerg-hhr').assets['zarr-abfs']
        ds      = xr.open_zarr(fsspec.get_mapper(assets.href,**assets.extra_fields['xarray:storage_options']),consolidated=True)
        logger.info('   Successfully retrieved IMERG')
        return ds
    except Exception:
        logger.exception('   Failed to retrieve IMERG')
        sys.exit(1)

def standardize(da):
    '''
    Purpose: Standardize the names, data types, and order of dimensions of an xr.DataArray.
    Args: 
    - da (xr.DataArray): input DataArray
    Returns: 
    - xr.DataArray: standardized DataArray
    '''
    dimnames   = {'latitude':'lat','longitude':'lon','level':'lev'}
    da         = da.rename({old:new for old,new in dimnames.items() if old in da.dims})
    targetdims = [dim for dim in ('lev','time','lat','lon') if dim in da.dims]
    extradims  = [dim for dim in da.dims if dim not in targetdims]
    da = da.drop_dims(extradims) if extradims else da
    for dim in da.dims:
        if dim=='time':
            if da.coords[dim].dtype.kind!='M':
                da.coords[dim] = da.indexes[dim].to_datetimeindex()
            da = da.sel(time=~da.time.to_index().duplicated(keep='first'))
        else:
            da.coords[dim] = da.coords[dim].astype('float32')
            if dim=='lon':
                da.coords[dim] = (da.coords[dim]+180)%360-180
    da = da.sortby(targetdims).transpose(*targetdims)   
    return da
    
def subset(da,years=YEARS,months=MONTHS,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE):
    '''
    Purpose: Subset an xr.DataArray by time (years/months), latitude/longitude ranges, and, if present, by pressure levels.
    Args:
    - da (xr.DataArray): input DataArray
    - years (list[int]): years to include (defaults to YEARS)
    - months (list[int]): months to include (defaults to MONTHS)
    - latrange (tuple[float,float]): minimum/maximum latitude (defaults to LATRANGE)
    - lonrange (tuple[float,float]): minimum/maximum longitude (defaults to LONRANGE)
    - levrange (tuple[float,float]): minimum/maximum pressure level in hPa (defaults to LEVRANGE, used only if 'lev' is a dimension)
    Returns:
    - xr.DataArray: subsetted DataArray
    ''' 
    da = da.sel(time=(da.time.dt.year.isin(years))&(da.time.dt.month.isin(months)))
    da = da.sel(lat=slice(*latrange),lon=slice(*lonrange))
    if 'lev' in da.dims:
        da = da.sel(lev=slice(*levrange))
    return da

def dataset(da,shortname,longname,units,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Wrap a standardized xr.DataArray into an xr.Dataset, preserving coordinates and setting variable and global metadata.
    Args:
    - da (xr.DataArray): input DataArray
    - shortname (str): variable name (abbreviation)
    - longname (str): variable long name/description
    - units (str): variable units
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)    
    Returns:
    - xr.Dataset: Dataset containing the variable named 'shortname' and metadata
    '''    
    ds = da.to_dataset(name=shortname)
    ds[shortname].attrs = dict(long_name=longname,units=units)
    ds.time.attrs = dict(long_name='Time')
    ds.lat.attrs  = dict(long_name='Latitude',units='°N')
    ds.lon.attrs  = dict(long_name='Longitude',units='°E')
    if 'lev' in ds.dims:
        ds.lev.attrs = dict(long_name='Pressure level',units='hPa')
    ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
    logger.info(f'   {longname} size: {ds.nbytes*1e-9:.2f} GB')
    return ds

def process(da,shortname,longname,units,years=YEARS,months=MONTHS,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Convert an xr.DataArray into an xr.Dataset by applying the standardize(), subset(), and dataset() functions in sequence.
    Args:
    - da (xr.DataArray): input DataArray
    - shortname (str): variable name (abbreviation)
    - longname (str): variable long name/description 
    - units (str): variable units
    - years (list[int]): years to include (defaults to YEARS)
    - months (list[int]): months to include (defaults to MONTHS)
    - latrange (tuple[float,float]): minimum/maximum latitude (defaults to LATRANGE)
    - lonrange (tuple[float,float]): minimum/maximum longitude (defaults to LONRANGE)
    - levrange (tuple[float,float]): minimum/maximum pressure level in hPa (defaults to LEVRANGE, used only if 'lev' is a dimension)
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)
    Returns:
    - xr.Dataset: processed Dataset
    ''' 
    da = standardize(da)
    da = subset(da,years,months,latrange,lonrange,levrange)
    ds = dataset(da,shortname,longname,units,author,email)
    return ds

def save(ds,savedir=SAVEDIR):
    '''
    Purpose: Save an xr.Dataset to a NetCDF file in the specified directory, then verify the write by reopening.
    Args:
    - ds (xr.Dataset): Dataset to save
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    os.makedirs(savedir,exist_ok=True)
    shortname = list(ds.data_vars)[0]
    longname  = ds[shortname].attrs['long_name']
    filename  = re.sub(r'\s+','_',longname)+'.nc'
    filepath  = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filename}...')   
    try:
        ds.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('   File write successful')
        return True
    except Exception:
        logger.exception('   Failed to save or verify')
        return False

if __name__=='__main__':
    try:
        logger.info('Retrieving ERA5 and IMERG data...')
        era5  = retrieve_era5()
        imerg = retrieve_imerg()
        logger.info('Extracting variable data...')
        prdata = imerg.precipitationCal.where((imerg.precipitationCal!=-9999.9)&(imerg.precipitationCal>=0),np.nan)*24
        psdata = era5.surface_pressure/100
        tdata  = era5.temperature
        qdata  = era5.specific_humidity
        del era5,imerg
        logger.info('Creating datasets...')
        dslist = [
            process(prdata,'pr','IMERG V06 precipitation rate','mm/day'),
            process(psdata,'ps','ERA5 surface pressure','hPa'),
            process(tdata,'t','ERA5 air temperature','K'),
            process(qdata,'q','ERA5 specific humidity','kg/kg')]
        del prdata,psdata,tdata,qdata
        logger.info('Saving datasets...')
        for ds in dslist:
            save(ds)
            del ds
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')