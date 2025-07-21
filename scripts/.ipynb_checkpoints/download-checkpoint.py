import os
import re
import gcsfs
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

def get_era5():
    '''
    Purpose: Access ERA5 data stored on Google Cloud Storage and return it as an xarray.Dataset.
    Returns: 
    - xarray.Dataset: ERA5 Dataset
    '''
    try:
        store = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
        ds = xr.open_zarr(store,decode_times=True)  
        logger.info('Successfully fetched ERA5')
        return ds
    except Exception as e:
        logger.error(f'Failed to fetch ERA5: {str(e)}')
        return None

def get_imerg():
    '''
    Purpose: Access IMERG V06 data stored on Microsoft Planetary Computer and return it as an xarray.Dataset.
    Returns: 
    - xarray.Dataset: IMERG V06 Dataset
    '''
    try:
        store   = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        catalog = pystac.Client.open(store, modifier=planetary_computer.sign_inplace)
        assets  = catalog.get_collection('gpm-imerg-hhr').assets['zarr-abfs']
        ds      = xr.open_zarr(fsspec.get_mapper(assets.href,**assets.extra_fields['xarray:storage_options']),consolidated=True)
        logger.info('Successfully fetched IMERG')
        return ds
    except Exception as e:
        logger.error(f'Failed to fetch IMERG: {str(e)}')
        return None

def standardize(da):
    '''
    Purpose: Standardize the dimensions and coordinates of an xarray.DataArray.
    Args: 
    - da (xarray.DataArray): input DataArray
    Returns: 
    - xarray.DataArray: standardized DataArray
    '''
    dimnames   = {'latitude':'lat','longitude':'lon','level':'lev'}
    da         = da.rename({oldname:newname for oldname,newname in dimnames.items() if oldname in da.dims})
    targetdims = ['lev','time','lat','lon'] if 'lev' in da.dims else ['time','lat','lon']
    extradims  = [dim for dim in da.dims if dim not in targetdims]
    if extradims:
        da = da.drop_dims(extradims)
    for dim in targetdims:
        if dim=='time':
            if da.coords[dim].dtype.kind!='M':
                da.coords[dim] = da.indexes[dim].to_datetimeindex()
            da = da.sel(time=~da.time.to_index().duplicated(keep='first'))
        elif dim=='lon':
            da.coords[dim] = (da.coords[dim]+180)%360-180        
        elif dim!='time':
            da.coords[dim] = da.coords[dim].astype(float)
    da = da.sortby(targetdims).transpose(*targetdims)   
    return da
    
def subset(da,years=YEARS,months=MONTHS,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE):
    '''
    Purpose: Subset an xarray.DataArray based on specified time, latitude, longitude, and pressure level ranges.
    Args:
    - da (xarray.DataArray): input DataArray
    - years (list): list of years to include (defaults to YEARS)
    - months (list): list of months to include (defaults to MONTHS)
    - latrange (tuple): minimum and maximum latitude (defaults to LATRANGE)
    - lonrange (tuple): minimum and maximum longitude (defaults to LONRANGE)
    - levrange (tuple): minimum and maximum pressure levels (defaults to LEVRANGE)    
    Returns:
    - xarray.DataArray: subsetted DataArray
    '''    
    da = da.sel(time=(da.time.dt.year.isin(years))&(da.time.dt.month.isin(months)))
    da = da.sel(lat=slice(*latrange),lon=slice(*lonrange))
    if 'lev' in da.dims:
        da = da.sel(lev=slice(*levrange))
    return da

def dataset(da,shortname,longname,units,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Format an xarray.DataArray into an xarray.Dataset with standard attributes and metadata.
    Args:
    - da (xarray.DataArray): input DataArray
    - shortname (str): variable name abbreviation
    - longname (str): variable name description
    - units (str): variable units
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)    
    Returns:
    - xarray.Dataset: formatted Dataset
    '''    
    ds = xr.Dataset(data_vars={shortname:([*da.dims],da.data)},coords={dim:da.coords[dim].data for dim in da.dims})
    ds[shortname].attrs = dict(long_name=longname,units=units)
    ds.time.attrs = dict(long_name='Time')
    ds.lat.attrs  = dict(long_name='Latitude',units='°N')
    ds.lon.attrs  = dict(long_name='Longitude',units='°E')
    if 'lev' in ds.dims:
        ds.lev.attrs = dict(long_name='Pressure level',units='hPa')
    ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
    logger.info(f'{shortname}: {ds.nbytes*1e-9:.2f} GB')
    return ds

def process(da,shortname,longname,units,years=YEARS,months=MONTHS,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Process an xarray.DataArray into an xarray.Dataset by applying the standardize(), subset(), and dataset() functions in sequence.
    Args:
    - da (xarray.DataArray): input DataArray
    - shortname (str): variable name abbreviation
    - longname (str): variable name description 
    - units (str): variable units
    - years (list): list of years to include (defaults to YEARS)
    - months (list): list of months to include (defaults to MONTHS)
    - latrange (tuple): minimum and maximum latitude (defaults to LATRANGE)
    - lonrange (tuple): minimum and maximum longitude (defaults to LONRANGE)
    - levrange (tuple): minimum and maximum pressure levels (defaults to LEVRANGE)
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)
    Returns:
    - xarray.Dataset: processed Dataset
    '''    
    da = standardize(da)
    da = subset(da,years,months,latrange,lonrange,levrange)
    ds = dataset(da,shortname,longname,units,author,email)
    return ds

def save(ds,savedir=SAVEDIR):
    '''
    Purpose: Save an xarray.Dataset to a NetCDF file in the specified directory. 
             Verify the file was saved successfully by attempting to reopen it.
    Args:
    - ds (xarray.Dataset): Dataset to save
    - savedir (str): directory where the file should be saved (defaults to SAVEDIR)
    Returns:
    - bool: True if the save operation was successful, False otherwise
    '''  
    shortname = list(ds.data_vars)[0]
    longname  = ds[shortname].attrs['long_name']
    filename  = re.sub(r'\s+','_',longname)+'.nc'
    filepath  = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filename}...')   
    try:
        ds.to_netcdf(filepath)
        logger.info(f'File written successfully: {filename}')
    except Exception as e:
        logger.error(f'Failed to write {filename}: {e}')
        return False
    try:
        with xr.open_dataset(filepath) as testds:
            logger.info(f'File verification successful: {filename}')
            return True
    except Exception as e:
        logger.warning(f'File saved but verification failed for {filename}: {e}')
        if os.path.exists(filepath):
            filesize = os.path.getsize(filepath)/(1024**3)
            logger.info(f'File exists with size {filesize:.2f} GB; considering save successful')
            return True
        else:
            logger.error(f'File does not exist after save: {filename}')
            return False

if __name__=='__main__':
    try:
        logger.info('Fetching ERA5 and IMERG data...')
        era5  = get_era5()
        imerg = get_imerg()
        logger.info('Extracting variable data...')
        pr = imerg.precipitationCal.where((imerg.precipitationCal!=-9999.9)&(imerg.precipitationCal>=0),np.nan)*24 # mm/hr to mm/day
        ps = era5.surface_pressure/100 # Pa to hPa
        t  = era5.temperature
        q  = era5.specific_humidity
        del era5,imerg
        logger.info('Processing and saving variables...')
        dslist = [
            process(pr,'pr','IMERG V06 precipitation rate','mm/day'),
            process(ps,'ps','ERA5 surface pressure','hPa'),
            process(t,'t','ERA5 air temperature','K'),
            process(q,'q','ERA5 specific humidity','kg/kg')]
        for ds in dslist:
            save(ds)
            del ds
        logger.info(f'Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')