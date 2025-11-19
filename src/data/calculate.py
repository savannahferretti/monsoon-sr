#!/usr/bin/env python

import os
import xesmf
import logging
import warnings
import numpy as np
import xarray as xr
from datetime import datetime

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

AUTHOR   = 'Savannah L. Ferretti'      
EMAIL    = 'savannah.ferretti@uci.edu' 
FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/raw'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/interim'
LATRANGE = (5.,25.) 
LONRANGE = (60.,90.)

def retrieve(longname,filedir=FILEDIR):
    '''
    Purpose: Lazily import in a NetCDF file as an xr.DataArray and, if applicable, ensure pressure levels are ascending (e.g., [500,550,600,...] hPa).
    Args:
    - longname (str): variable long name/description
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - xr.DataArray: loaded DataArray with levels ordered (if applicable) 
    '''
    filename = f'{longname}.nc'
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath,engine='h5netcdf')
    if 'lev' in da.dims:
        if not np.all(np.diff(da['lev'].values)>0):
            da = da.sortby('lev')
            logger.info(f'   Levels for {filename} were reordered to ascending')
    return da
    
def create_p_array(refda):
    '''
    Purpose: Create a pressure xr.DataArray from the 'lev' dimension.
    Args:
    - refda (xr.DataArray): reference DataArray containing 'lev'
    Returns:
    - xr.DataArray: pressure DataArray
    '''
    p = refda.lev.expand_dims({'time':refda.time,'lat':refda.lat,'lon':refda.lon}).transpose('lev','time','lat','lon')
    return p

def create_level_mask(refda,ps):
    '''
    Purpose: Create a below-surface level mask; 1 where levels exist (lev ≤ ps), else 0.
    - refda (xr.DataArray): reference DataArray containing 'lev'
    - ps (xr.DataArray): surface pressure (hPa)
    Returns:
    - xr.DataArray: DataArray of 0's (invalid levels) or 1's (valid levels)
    '''
    levmask = (refda.lev<=ps).transpose('time','lat','lon','lev').astype('uint8')
    return levmask

def resample(da):
    '''
    Purpose: Compute a centered hourly mean (uses the two half-hour samples that straddle each hour; falls back to 
    one at boundaries).
    Args:
    - da (xr.DataArray): input DataArray
    Returns:
    - xr.DataArray: DataArray resampled at on-the-hour timestamps
    '''
    da = da.rolling(time=2,center=True,min_periods=1).mean()
    da = da.sel(time=da.time.dt.minute==0)
    return da
    
def regrid(da,latrange=LATRANGE,lonrange=LONRANGE):
    '''
    Purpose: Regrids a DataArray to a 1° target grid.
    Args:
    - da (xr.DataArray): input DataArray (with halo)
    - latrange (tuple[float,float]): target latitude range (defaults to LATRANGE)
    - lonrange (tuple[float,float]): target longitude range (defaults to LONRANGE)
    Returns:
    - xr.DataArray: DataArray regridded to target domain
    '''
    targetlats = np.arange(latrange[0],latrange[1]+1,1.)
    targetlons = np.arange(lonrange[0],lonrange[1]+1,1.)
    targetgrid = xr.Dataset({'lat':(['lat'],targetlats),'lon':(['lon'],targetlons)})
    regridder  = xesmf.Regridder(da,targetgrid,method='conservative')
    da = regridder(da,keep_attrs=True)
    return da
    
def calc_es(t):
    '''
    Purpose: Calculate saturation vapor pressure (eₛ) using Eqs. 17 and 18 from Huang J. (2018), J. Appl. Meteorol. Climatol.
    Args:
    - t (xr.DataArray): temperature DataArray (K) 
    Returns:
    - xr.DataArray: eₛ DataArray (hPa)
    '''    
    tc = t-273.15
    eswat = np.exp(34.494-(4924.99/(tc+237.1)))/((tc+105.)**1.57)
    esice = np.exp(43.494-(6545.8/(tc+278.)))/((tc+868.)**2.)
    es = xr.where(tc>0.,eswat,esice)/100.
    return es

def calc_qs(p,t):
    '''
    Purpose: Calculate saturation specific humidity (qₛ) using Eq. 4 from Miller SFK. (2018), Atmos. Humidity Eq. Plymouth State Wea. Ctr.
    Args:
    - p (xr.DataArray): pressure DataArray (hPa)
    - t (xr.DataArray): temperature DataArray (K)
    Returns:
    - xr.DataArray: qₛ DataArray (kg/kg)
    '''
    rv = 461.50   
    rd = 287.04   
    epsilon = rd/rv
    es = calc_es(t) 
    qs = (epsilon*es)/(p-es*(1.-epsilon))
    return qs

def calc_thetae(p,t,q=None,ps=None):
    '''
    Purpose: Calculate equivalent potential temperature (θₑ) using Eqs. 43 and 55 from Bolton D. (1980), Mon. Wea. Rev. 
    Options to calculate θₑ at the surface, or the saturated θₑ, are given.        
    Args:
    - p (xr.DataArray): pressure DataArray (hPa)
    - t (xr.DataArray): temperature DataArray (K)
    - q (xr.DataArray, optional): specific humidity DataArray (kg/kg); if None, saturated θₑ will be calculated
    - ps (xr.DataArray, optional): surface pressure DataArray (hPa); if given, θₑ at the surface will be calculated
      (ps > 1,000 hPa are clamped to 1,000 hPa to prevent extrapolation beyond the available pressure levels from ERA5)
    Returns:
    - xr.DataArray: (regular, surface, or saturated) θₑ DataArray (K)
    '''
    if q is None:
        q = calc_qs(p,t)
    if ps is not None:
        psclamped = xr.where(ps>1000.,1000.,ps)
        t = t.interp(lev=psclamped)
        q = q.interp(lev=psclamped)
        p = psclamped
    p0 = 1000.  
    rv = 461.5  
    rd = 287.04
    epsilon = rd/rv
    r  = q/(1.-q) 
    e  = (p*r)/(epsilon+r)
    tl = 2840./(3.5*np.log(t)-np.log(e)-4.805)+55.
    thetae = t*(p0/p)**(0.2854*(1.-0.28*r))*np.exp((3.376/tl-0.00254)*1000.*r*(1.+0.81*r))
    return thetae

def get_level_above(ptarget,levels,side):
    '''
    Purpose: Find the pressure level immediately above a target pressure, i.e., the next smallest pressure (higher altitude).
    Args:
    - ptarget (xr.DataArray or np.ndarray): target pressures
    - levels (np.ndarray): 1D array of ascending pressure levels (e.g., [500,550,600,...] hPa)
    - side (str): 'left' or 'right' tie-breaking for np.searchsorted
    Returns:
    - np.ndarray: array of pressure levels immediately above each target (same shape as 'ptarget')
    '''
    searchidx = np.searchsorted(levels,ptarget,side=side)
    levabove  = levels[np.maximum(searchidx-1,0)]
    return levabove

def get_level_below(ptarget,levels,side):
    '''
    Purpose: Find the pressure level immediately below a target pressure, i.e., the next largest pressure (lower altitude).
    Args:
    - ptarget (xr.DataArray or np.ndarray): target pressures
    - levels (np.ndarray): 1D array of ascending pressure levels (e.g., [500,550,600,...] hPa)
    - side (str): 'left' or 'right' tie-breaking for np.searchsorted
    Returns:
    - np.ndarray: array of pressure levels immediately below each target (same shape as 'ptarget')
    '''
    searchidx = np.searchsorted(levels,ptarget,side=side)
    levbelow  = levels[np.minimum(searchidx,len(levels)-1)]
    return levbelow

def calc_layer_average(da,a,b):
    '''
    Purpose: Calculate the pressure-weighted mean of an xr.DataArray between two pressure levels 'a' (bottom of layer) and 
    'b' (top of layer), with `a > b`.
    Args:
    - da (xr.DataArray): input DataArray with 'lev' dimension
    - a (xr.DataArray): DataArray of bottom boundary pressures (higher values, lower altitude)
    - b (xr.DataArray): DataArray of top boundary pressures (lower values, higher altitude)
    Returns:
    - xr.DataArray: layer-averaged DataArray
    '''
    da = da.load()
    a  = a.load()
    b  = b.load()
    levabove = xr.apply_ufunc(get_level_above,a,kwargs={'levels':np.array(da.lev),'side':'right'})
    levbelow = xr.apply_ufunc(get_level_below,a,kwargs={'levels':np.array(da.lev),'side':'right'})
    valueabove = da.sel(lev=levabove)
    valuebelow = da.sel(lev=levbelow)
    correction = -valueabove/2*(levbelow-levabove)*(a<da.lev[-1])
    levbelow   = levbelow+(levbelow==levabove)
    lowerintegral = (a-levabove)*valueabove+(valuebelow-valueabove)*(a-levabove)**2/(levbelow-levabove)/2+correction
    lowerintegral = lowerintegral.fillna(0)
    innerintegral = (da*(da.lev<=a)*(da.lev>=b)).fillna(0).integrate('lev')
    levabove = xr.apply_ufunc(get_level_above,b,kwargs={'levels':np.array(da.lev),'side':'left'})
    levbelow = xr.apply_ufunc(get_level_below,b,kwargs={'levels':np.array(da.lev),'side':'left'})
    valueabove = da.sel(lev=levabove)
    valuebelow = da.sel(lev=levbelow)
    correction = -valuebelow/2*(levbelow-levabove)*(b>da.lev[0])
    levabove   = levabove-(levbelow==levabove)
    upperintegral = (levbelow-b)*valueabove+(valuebelow-valueabove)*(levbelow-levabove)*(1-((b-levabove)/(levbelow-levabove))**2)/2+correction
    upperintegral = upperintegral.fillna(0)  
    layeraverage  = (lowerintegral+innerintegral+upperintegral)/(a-b)
    return layeraverage
    
def calc_weights(ps,pbltop,lfttop):
    '''
    Purpose: Calculate weights for the boundary layer (PBL) and lower free troposphere (LFT) using Eqs. 5a and 5b from Adames AF, 
    Ahmed F, and Neelin JD. 2021. J. Atmos. Sci.
    Args:
    - ps (xr.DataArray): surface pressure DataArray (hPa)
    - pbltop (xr.DataArray): DataArray of pressures at the top of the PBL (hPa)
    - lfttop (xr.DataArray): DataArray of pressures at the top of the LFT (hPa)
    Returns:
    - tuple[xr.DataArray,xr.DataArray]: PBL and LFT weights
    '''
    pblthickness = ps-pbltop
    lftthickness = pbltop-lfttop
    wb = (pblthickness/lftthickness)*np.log((pblthickness+lftthickness)/pblthickness)
    wl = 1-wb
    return wb,wl

def calc_bl_terms(thetaeb,thetael,thetaelsat,wb,wl):
    '''
    Purpose: Calculate CAPEL, SUBSATL, and BL following Eq. 1 from Ahmed F and Neelin JD. 2021. Geophys. Res. Lett.
    Args:
    - thetaeb (xr.DataArray): DataArray of θₑ averaged over the PBL (K)
    - thetael (xr.DataArray): DataArray of θₑ averaged over the LFT (K)
    - thetaelsat (xr.DataArray): DataArray of saturated θₑ averaged over the LFT (K)
    - wb (xr.DataArray): DataArray of PBL weights
    - wl (xr.DataArray): DataArray of LFT weights
    Returns:
    - tuple[xr.DataArray,xr.DataArray,xr.DataArray]: CAPEL, SUBSATL, and BL DataArrays
    '''
    g       = 9.81
    kappal  = 3.
    thetae0 = 340.
    cape    = ((thetaeb-thetaelsat)/thetaelsat)*thetae0
    subsat  = ((thetaelsat-thetael)/thetaelsat)*thetae0
    bl      = (g/(kappal*thetae0))*((wb*cape)-(wl*subsat))
    return cape,subsat,bl

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
    dims = [dim for dim in ('time','lat','lon','lev') if dim in da.dims]
    da = da.transpose(*dims)
    ds = da.to_dataset(name=shortname)
    ds[shortname].attrs = dict(long_name=longname,units=units)
    if 'time' in ds.coords:
        ds.time.attrs = dict(long_name='Time')
    if 'lat' in ds.coords:
        ds.lat.attrs  = dict(long_name='Latitude',units='°N')
    if 'lon' in ds.coords:
        ds.lon.attrs  = dict(long_name='Longitude',units='°E')
    if 'lev' in ds.coords:
        ds.lev.attrs  = dict(long_name='Pressure level',units='hPa')
    ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
    logger.info(f'   {shortname}: {ds.nbytes*1e-9:.3f} GB')
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
    filename  = f'{shortname}.nc' 
    filepath  = os.path.join(savedir,filename)
    encoding  = {name: {'dtype':('uint8' if name=='levmask' else 'float32')} for name in ds.data_vars}
    encoding.update({coord:{'dtype':'float32'} for coord in ('lat','lon','lev') if coord in ds.coords})
    logger.info(f'   Attempting to save {filename}...')   
    try:
        ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    try:
        logger.info('Importing all raw variables...')
        pr = retrieve('IMERG_V06_precipitation_rate')
        lf = retrieve('ERA5_land_fraction')
        ps = retrieve('ERA5_surface_pressure')
        t  = retrieve('ERA5_air_temperature')
        q  = retrieve('ERA5_specific_humidity')
        logger.info('Resampling/regridding variables...')
        pr = regrid(resample(pr)).clip(min=0).load()
        lf = regrid(lf).clip(0,1).load()
        ps = regrid(ps).load()
        t  = regrid(t).load()
        q  = regrid(q).load()
        logger.info('Creating below-surface level mask...')
        levmask = create_level_mask(t,ps)
        logger.info('Calculating equivalent potential temperature terms...')
        p          = create_p_array(q)
        thetae     = calc_thetae(p,t,q)
        thetaesat  = calc_thetae(p,t)
        thetaesurf = calc_thetae(p,t,q,ps)
        logger.info('Calculating CAPE-like and SUBSAT-like proxy terms...')
        capeproxy   = thetaesurf-thetaesat
        subsatproxy = thetaesat-thetae
        logger.info('Calculating layer averages...')
        pbltop     = ps-100.
        lfttop     = xr.full_like(ps,500.)
        thetaeb    = calc_layer_average(thetae,ps,pbltop)*np.sqrt(-1+2*(ps>lfttop))
        thetael    = calc_layer_average(thetae,pbltop,lfttop)
        thetaelsat = calc_layer_average(thetaesat,pbltop,lfttop)
        wb,wl      = calc_weights(ps,pbltop,lfttop)
        logger.info('Calculating BL terms...')
        cape,subsat,bl = calc_bl_terms(thetaeb,thetael,thetaelsat,wb,wl)
        logger.info('Creating datasets...')
        dslist = [
            dataset(pr,'pr','Precipitation rate','mm/hr'),
            dataset(lf,'lf','Land fraction','0-1'),
            dataset(bl,'bl','Average buoyancy in the lower troposphere','m/s²'),
            dataset(cape,'cape','Undilute buoyancy in the lower troposphere','K'),
            dataset(subsat,'subsat', 'Lower free-tropospheric subsaturation','K'),
            dataset(capeproxy,'capeproxy','θₑ(surface) - saturated θₑ(p)','K'),
            dataset(subsatproxy,'subsatproxy','Saturated θₑ(p) - θₑ(p)','K'),
            dataset(t,'t','Air temperature','K'),
            dataset(q,'q','Specific humidity','kg/kg'),
            dataset(levmask,'levmask','Below-surface level mask','N/A')]
        logger.info('Saving datasets...')
        for ds in dslist:
            save(ds)
            del ds
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')