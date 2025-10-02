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

AUTHOR    = 'Savannah L. Ferretti'      
EMAIL     = 'savannah.ferretti@uci.edu' 
FILEDIR   = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/raw'
SAVEDIR   = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/interim'

def retrieve(longname,filedir=FILEDIR):
    '''
    Purpose: Lazily import a NetCDF file as an xr.DataArray and, if applicable, ensure pressure levels are ascending (e.g., [500,550,600,...] hPa).
    Args:
    - longname (str): variable long name/description
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - xr.DataArray: DataArray with levels ordered (if applicable) 
    '''
    filename = f'{longname}.nc'
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath,engine='h5netcdf')
    if 'lev' in da.dims:
        if not np.all(np.diff(da['lev'].values)>0):
            da = da.sortby('lev')
            logger.info(f'   Levels for {filename} were reordered to ascending')
    return da
    
def create_p_array(da):
    '''
    Purpose: Create a pressure xr.DataArray from the 'lev' dimension.
    Args:
    - da (xr.DataArray): DataArray containing 'lev'
    Returns:
    - xr.DataArray: pressure DataArray
    '''
    p = da.lev.expand_dims({'time':da.time,'lat':da.lat,'lon':da.lon}).transpose('lev','time','lat','lon')
    return p

def regrid_and_resample(da,gridtarget,method='conservative_normed'):
    '''
    Purpose: Compute a centered hourly mean (uses the two half-hour samples that straddle each hour; falls back to 
    one at boundaries) and then regrid to a target latitude–longitude grid.
    Args:
    - da (xr.DataArray): input DataArray
    - gridtarget (xr.DataArray): DataArray with target 'lat' and 'lon' for regridding
    - method (str): 'bilinear' | 'conservative' | 'conservative_normed' | 'patch' | 'nearest_s2d' | 'nearest_d2s' (defaults to 'conservative_normed')
    Returns:
    - xr.DataArray: DataArray regridded to the target grid at on-the-hour timestamps
    '''
    da = da.rolling(time=2,center=True,min_periods=1).mean()
    da = da.sel(time=da.time.dt.minute==0)
    regridder = xesmf.Regridder(da,gridtarget,method=method)
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
    dims = ('time','lat','lon','lev') if 'lev' in da.dims else ('time','lat','lon')
    da = da.transpose(*dims)
    ds = da.to_dataset(name=shortname)
    ds[shortname].attrs = dict(long_name=longname,units=units)
    ds.time.attrs = dict(long_name='Time')
    ds.lat.attrs = dict(long_name='Latitude',units='°N')
    ds.lon.attrs = dict(long_name='Longitude',units='°E')
    if 'lev' in ds.dims:
        ds.lev.attrs = dict(long_name='Pressure level',units='hPa')
    ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
    logger.info(f'{shortname}: {ds.nbytes*1e-9:.2f} GB')
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
    encoding  = (
        {vardata:{'dtype':'float32'} for vardata in ds.data_vars}
        | {coord:{'dtype':'float32'} for coord in ('lat','lon','lev') if coord in ds.coords})
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
        ps = retrieve('ERA5_surface_pressure')
        t  = retrieve('ERA5_air_temperature')
        q  = retrieve('ERA5_specific_humidity')
        logger.info('Resampling/regridding precipitation...')
        resampledpr = regrid_and_resample(pr.load(),ps.load())
        del pr
        logger.info('Beginning chunking...')
        nyears     = len(np.unique(ps.time.dt.year.values))
        timechunks = np.array_split(np.arange(len(ps.time)),nyears)
        results    = {
            'bl':[],
            'cape':[],
            'subsat':[],
            'capeproxy':[],
            'subsatproxy':[],
            't':[],
            'q':[]}
        for i,timechunk in enumerate(timechunks):
            logger.info(f'Processing time chunk {i+1}/{len(timechunks)}...')
            pschunk = ps.isel(time=timechunk).load()
            tchunk  = t.isel(time=timechunk).load()
            qchunk  = q.isel(time=timechunk).load()
            pchunk  = create_p_array(qchunk)
            logger.info('   Calculating equivalent potential temperature terms')
            thetaechunk     = calc_thetae(pchunk,tchunk,qchunk)
            thetaesatchunk  = calc_thetae(pchunk,tchunk)
            thetaesurfchunk = calc_thetae(pchunk,tchunk,qchunk,pschunk)
            del pchunk        
            logger.info('   Calculating CAPE-like and SUBSAT-like proxy terms')    
            capeproxychunk   = thetaesurfchunk-thetaesatchunk
            subsatproxychunk = thetaesatchunk-thetaechunk
            del thetaesurfchunk
            logger.info('   Calculating layer averages')
            pbltopchunk = pschunk-100.
            lfttopchunk = xr.full_like(pschunk,500.)
            thetaebchunk    = calc_layer_average(thetaechunk,pschunk,pbltopchunk)*np.sqrt(-1+2*(pschunk>lfttopchunk))
            thetaelchunk    = calc_layer_average(thetaechunk,pbltopchunk,lfttopchunk)
            thetaelsatchunk = calc_layer_average(thetaesatchunk,pbltopchunk,lfttopchunk)
            wbchunk,wlchunk = calc_weights(pschunk,pbltopchunk,lfttopchunk)
            del pschunk,pbltopchunk,lfttopchunk,thetaechunk,thetaesatchunk
            logger.info('   Calculating BL terms')
            capechunk,subsatchunk,blchunk = calc_bl_terms(thetaebchunk,thetaelchunk,thetaelsatchunk,wbchunk,wlchunk)
            del wbchunk,wlchunk,thetaebchunk,thetaelchunk,thetaelsatchunk
            logger.info('   Appending chunk results')
            results['bl'].append(blchunk)
            results['cape'].append(capechunk)
            results['subsat'].append(subsatchunk)
            results['capeproxy'].append(capeproxychunk)
            results['subsatproxy'].append(subsatproxychunk)
            results['t'].append(tchunk)
            results['q'].append(qchunk)
            del blchunk,capechunk,subsatchunk,capeproxychunk,subsatproxychunk,tchunk,qchunk
        del ps,t,q
        logger.info('Creating datasets...')
        dslist = [
            dataset(resampledpr,'pr','Resampled/regridded precipitation rate','mm/hr'),
            dataset(xr.concat(results['bl'],dim='time'),'bl','Average buoyancy in the lower troposphere','m/s²'),
            dataset(xr.concat(results['cape'],dim='time'),'cape','Undilute buoyancy in the lower troposphere','K'),
            dataset(xr.concat(results['subsat'],dim='time'),'subsat','Lower free-tropospheric subsaturation','K'),
            dataset(xr.concat(results['capeproxy'],dim='time'),'capeproxy','θₑ(surface) - saturated θₑ(p)','K'),
            dataset(xr.concat(results['subsatproxy'],dim='time'),'subsatproxy','Saturated θₑ(p) - θₑ(p)','K'), 
            dataset(xr.concat(results['t'],dim='time'),'t','Air temperature','K'),
            dataset(xr.concat(results['q'],dim='time'),'q','Specific humidity','kg/kg')]
        logger.info('Saving datasets...')
        for ds in dslist:
            save(ds)
            del ds
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')