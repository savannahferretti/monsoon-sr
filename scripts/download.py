import os
import gcsfs
import fsspec
import warnings
import numpy as np
import xarray as xr
import planetary_computer
from datetime import datetime
import pystac_client as pystac
from multiprocessing import Pool

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
    store = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    ds    = xr.open_zarr(store,decode_times=True)  
    return ds

def get_imerg():
    store   = 'https://planetarycomputer.microsoft.com/api/stac/v1'
    catalog = pystac.Client.open(store,modifier=planetary_computer.sign_inplace)
    assets  = catalog.get_collection('gpm-imerg-hhr').assets['zarr-abfs']
    ds      = xr.open_zarr(fsspec.get_mapper(assets.href,**assets.extra_fields['xarray:storage_options']),consolidated=True)
    return ds

def standardize(da):
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
    ds = ds.sel(time=(ds['time.year'].isin(years))&(ds['time.month'].isin(months)))
    ds = ds.sel(lat=slice(*latrange),lon=slice(*lonrange))
    if 'lev' in ds.dims:
        ds = ds.sel(lev=slice(*levrange))
    return ds
    
def preprocess(da,shortname,longname,units,years=YEARS,months=MONTHS,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE,author=AUTHOR,email=EMAIL):
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
    print(f'{longname}: {ds.nbytes*1e-9:.2f} GB')
    return ds

def save(args,savedir=SAVEDIR):
    ds,filename = args
    filepath    = f'{savedir}/{filename}'
    ds.to_netcdf(filepath)
    print(f'Saved {filename}')

if __name__=='__main__':
    era5  = get_era5()
    imerg = get_imerg()

    prdata = imerg.precipitationCal.where((imerg.precipitationCal!=-9999.9)&(imerg.precipitationCal>=0),np.nan)*24 # mm/hr to mm/day
    psdata = era5.surface_pressure/100 # Pa to hPa
    qdata  = era5.specific_humidity
    tdata  = era5.temperature

    pr = preprocess(prdata,shortname='pr',longname='IMERG V06 precipitation rate',units='mm/day')
    ps = preprocess(psdata,shortname='ps',longname='ERA5 surface pressure',units='hPa')
    q  = preprocess(qdata,shortname='q',longname='ERA5 specific humidity',units='kg/kg')
    t  = preprocess(tdata,shortname='t',longname='ERA5 air temperature',units='K')

    # Prepare arguments for parallel processing
    saveargs = [(pr,'IMERG_precipitation_rate.nc'),
                (ps,'ERA5_surface_pressure.nc'),
                (q,'ERA5_specific_humidity.nc'),
                (t,'ERA5_temperature.nc')]

    # Use multiprocessing to save files in parallel
    with Pool(processes=4) as pool:
        pool.map(save,saveargs)

    print('All files saved successfully!')