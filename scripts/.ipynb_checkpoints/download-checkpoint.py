import os
import gcsfs
import fsspec
import logging
import warnings
import numpy as np
import xarray as xr
import planetary_computer
from datetime import datetime
import pystac_client as pystac
from multiprocessing import Pool

AUTHOR   = os.environ['AUTHOR']
EMAIL    = os.environ['EMAIL']
SAVEDIR  = os.environ['SAVEDIR']
YEARS    = [int(year) for year in os.environ['YEARS'].split()]
MONTHS   = [int(month) for month in os.environ['MONTHS'].split()]
LATRANGE = tuple(float(lat) for lat in os.environ['LATRANGE'].split())
LONRANGE = tuple(float(lon) for lon in os.environ['LONRANGE'].split())
LEVRANGE = tuple(float(lev) for lev in os.environ['LEVRANGE'].split())

# AUTHOR    = 'Savannah L. Ferretti'
# EMAIL     = 'savannah.ferretti@uci.edu'
# SAVEDIR   = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/raw'
# YEARS     = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
# MONTHS    = [6,7,8]
# LATRANGE  = (5.,25.) 
# LONRANGE  = (60.,90.)
# LEVRANGE  = (500.,1000.)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',filename='download.log',filemode='w')
logger = logging.getLogger()

def get_era5():
    '''
    Purpose: Accesses ERA5 data stored on Google Cloud Storage and returns it for further processing.
    Returns: 
    - ds (xarray.Dataset): ERA5 data
    '''
    store = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    ds    = xr.open_zarr(store,decode_times=True)  
    return ds

def get_imerg():
    '''
    Purpose: Accesses IMERG V06 data stored on Microsoft Planetary Computer and returns it for further processing.
    Returns: 
    - ds (xarray.Dataset): IMERG V06 data
    '''
    store   = 'https://planetarycomputer.microsoft.com/api/stac/v1'
    catalog = pystac.Client.open(store,modifier=planetary_computer.sign_inplace)
    assets  = catalog.get_collection('gpm-imerg-hhr').assets['zarr-abfs']
    ds      = xr.open_zarr(fsspec.get_mapper(assets.href,**assets.extra_fields['xarray:storage_options']),consolidated=True)
    return ds

def standardize(da):
    '''
    Purpose: Standardizes the dimensions and coordinates of an xarray.DataArray.
    Args: 
    - da (xarray.DataArray): input DataArray
    Returns: 
    - da (xarray.DataArray): standardized DataArray
    '''
    dimnames = {'latitude':'lat','longitude':'lon','level':'lev'}
    da = da.rename({oldname:newname for oldname,newname in dimnames.items() if oldname in da.dims})
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
    
def subset(ds,years=YEARS,months=MONTHS,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE):
    '''
    Purpose: Subsets an xarray.Dataset based on specified time, latitude, longitude, and level ranges.
    Args:
    - ds (xarray.Dataset): input Dataset
    - years (list): list of years to include
    - months (list): list of months to include
    - latrange (tuple): (minimum latitude, maximum latitude)
    - lonrange (tuple): (minimum longitude, maximum longitude)
    - levrange (tuple): (minimum pressure level, maximum pressure level)
    Returns:
    - ds (xarray.Dataset): subsetted Dataset
    '''    
    ds = ds.sel(time=(ds['time.year'].isin(years))&(ds['time.month'].isin(months)))
    ds = ds.sel(lat=slice(*latrange),lon=slice(*lonrange))
    if 'lev' in ds.dims:
        ds = ds.sel(lev=slice(*levrange))
    return ds

def preprocess(da,shortname,longname,units,years=YEARS,months=MONTHS,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Preprocesses an xarray.DataArray into a standardized Dataset.
    Args:
    - da (xarray.DataArray): input DataArray
    - shortname (str): variable name abbreviation
    - longname (str): full variable name
    - units (str): variable units
    - years (list): list of years to include
    - months (list): list of months to include
    - latrange (tuple): (minimum latitude, maximum latitude)
    - lonrange (tuple): (minimum longitude, maximum longitude)
    - levrange (tuple): (minimum pressure level, maximum pressure level)
    - author (str): author name for metadata
    - email (str): author email for metadata
    Returns:
    - ds (xarray.Dataset): preprocessed Dataset
    '''    
    logger.info(f'Preprocessing {longname}...')
    da = standardize(da)
    da = subset(da,years,months,latrange,lonrange,levrange)
    ds = xr.Dataset(data_vars={shortname:([*da.dims],da.data)},
                    coords={dim:da.coords[dim].data for dim in da.dims})
    ds[shortname].attrs = dict(long_name=longname,units=units)
    ds.time.attrs = dict(long_name='Time')
    ds.lat.attrs  = dict(long_name='Latitude',units='°N')
    ds.lon.attrs  = dict(long_name='Longitude',units='°E')
    if 'lev' in ds.dims:
        ds.lev.attrs = dict(long_name='Pressure level',units='hPa')
    ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
    logger.info(f'{longname}: {ds.nbytes*1e-9:.2f} GB')
    return ds

def save(args,savedir=SAVEDIR):
    '''
    Purpose: Saves an xarray.Dataset to a NetCDF file in the specified directory. It's designed to be used with 
    multiprocessing for parallel saving of multiple datasets.
    Args:
    - args (tuple): (xarray.Dataset, filename)
    - savedir (str): directory where the file should be saved
    '''    
    ds,filename = args
    filepath    = f'{savedir}/{filename}'
    logger.info(f'Saving {filename}...')
    try:
        ds.to_netcdf(filepath)
        logger.info(f'Successfully saved {filename}')
    except Exception as e:
        logger.error(f'Error saving {filename}: {str(e)}')

if __name__=='__main__':
    logger.info('Starting data download and processing...')
    try:
        logger.info('Fetching ERA5 data...')
        era5  = get_era5()
        logger.info('Fetching IMERG data...')
        imerg = get_imerg()
        logger.info('Pulling out indvidiual variables...')
        prdata = imerg.precipitationCal.where((imerg.precipitationCal!=-9999.9)&(imerg.precipitationCal>=0),np.nan)*24 # mm/hr to mm/day
        psdata = era5.surface_pressure/100 # Pa to hPa
        qdata  = era5.specific_humidity
        tdata  = era5.temperature
        logger.info('Preprocessing indvidiual variables...')
        pr = preprocess(prdata,shortname='pr',longname='IMERG V06 precipitation rate',units='mm/day')
        ps = preprocess(psdata,shortname='ps',longname='ERA5 surface pressure',units='hPa')
        q  = preprocess(qdata,shortname='q',longname='ERA5 specific humidity',units='kg/kg')
        t  = preprocess(tdata,shortname='t',longname='ERA5 air temperature',units='K')
        logger.info('Starting parallel file saving...')
        saveargs = [(pr,'IMERG_precipitation_rate.nc'),
                    (ps,'ERA5_surface_pressure.nc'),
                    (q,'ERA5_specific_humidity.nc'),
                    (t,'ERA5_temperature.nc')]
        with Pool(processes=4) as pool:
            pool.map(save,saveargs)
        logger.info('All files saved successfully!')
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
logger.info('Script execution completed!')