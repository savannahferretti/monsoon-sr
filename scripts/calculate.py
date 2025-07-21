import xesmf
import logging
import warnings
import numpy as np
import xarray as xr
import planetary_computer
from datetime import datetime

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

AUTHOR  = 'Savannah L. Ferretti'      
EMAIL   = 'savannah.ferretti@uci.edu' 
FILEDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-pod/data/raw'
SAVEDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-pod/data/interim'

def load(filename,filedir=FILEDIR):
    filepath = os.path.join(filedir,filename)
    ds = xr.open_dataarray(filepath)
    return ds.load()

def regrid(da,gridtarget):
    regridder = xesmf.Regridder(da,gridtarget,method='bilinear')
    regridded = regridder(da,keep_attrs=True)
    return regridded
  
def resample(da,frequency='H',method='first'):
    da.coords['time'] = da.time.dt.floor(frequency) 
    resampled         = da.groupby('time').mean()
    return resampled

pr = load('IMERG_V06_precipitation.nc')
ps = load('ERA5_surface_pressure.nc')
q  = load('ERA5_specific_humidity.nc')
t  = laod('ERA5_air_temperature.nc')

# Regrid/resample precipitation
pr = regrid(pr,ps)
pr = resample(pr)

# Filtered T and q (for pressure levels above ps)


# Calculate θe, θe*, θesurf


# Calculate layer averaged for BL terms



# Calculate BL terms



def save(ds,savedir=SAVEDIR):
    '''
    Purpose: Saves an xarray.Dataset to a NetCDF file in the specified directory. It records whether saving was a success or failure.
    Args:
    - ds (xarray.Dataset): processed variable Dataset to save
    - savedir (str): directory where the file should be saved (defaults to SAVEDIR)
    Returns:
    - bool: True if the save operation was successful, False otherwise
    '''  
    varname  = list(ds.data_vars)[0]
    longname = ds[varname].attrs['long_name']
    filename = re.sub(r'\s+','_',longname)+'.nc'
    filepath = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filepath}') 
    try:
        ds.to_netcdf(filepath)
        logger.info(f'Successfully saved {filename}')
        return True
    except Exception as e:
        logger.error(f'Failed to save {filename}: {str(e)}')
        return False

if __name__ == '__main__':
    try:
        logger.info('Loading data...')
        logger.info('Resampling/regridding precipitation...')
        logger.info('Filtering temperature/specific humidity levels...')
        logger.info('Calculating theta-e...')
        logger.info('Calculating theta-e,surf...')
        logger.info('Calculating saturated theta-e...')
        logger.info('Calculating layer averages...')
        logger.info('Calculating BL terms...')
        logger.info('Saving variables...')
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')