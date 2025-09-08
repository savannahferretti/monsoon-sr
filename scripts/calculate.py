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

def import_data(filename,filedir=FILEDIR):
    '''
    Purpose: Lazily import a NetCDF file as an xarray.DataArray and ensure pressure levels are ascending (if applicable).
    Args:
    - filename (str): file name to import
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - xarray.DataArray: DataArray with levels ordered (if applicable) 
    '''
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath,engine='h5netcdf')
    if 'lev' in da.dims:
        if not np.all(np.diff(da['lev'].values)>0):
            da = da.sortby('lev')
            logger.info(f'   Levels for {filename} were reordered to ascending')
    return da
    
def get_p_array(da):
    '''
    Purpose: Create a pressure xarray.DataArray from the 'lev' dimension.
    Args:
    - da (xarray.DataArray): DataArray containing 'lev'
    Returns:
    - xarray.DataArray: pressure DataArray
    '''
    p = da.lev.expand_dims({'time':da.time,'lat':da.lat,'lon':da.lon}).transpose('lev','time','lat','lon')
    return p

def regrid_and_resample(da,gridtarget,frequency='h'):
    '''
    Purpose: Regrid an xarray.DataArray to a target latitude-longitude grid and then resample in time.
    Args:
    - da (xarray.DataArray): input DataArray
    - gridtarget (xarray.DataArray): DataArray with target 'lat' and 'lon' for regridding
    - frequency (str): resampling frequency (defaults to 'h' for hourly)
    Returns:
    - xarray.DataArray: regridded and time-resampled DataArray
    '''
    regridder = xesmf.Regridder(da,gridtarget,method='bilinear')
    da = regridder(da,keep_attrs=True)
    da.coords['time'] = da.time.dt.floor(frequency) 
    da = da.groupby('time').first()
    return da
    
def calc_es(t):
    '''
    Purpose: Calculate saturation vapor pressure (eₛ) using Eqs. 17 and 18 from Huang J. 2018. J. Appl. Meteorol. Climatol.
    Args:
    - t (xarray.DataArray): temperature DataArray (K) 
    Returns:
    - xarray.DataArray: eₛ DataArray (hPa)
    '''    
    tc = t-273.15
    eswat = np.exp(34.494-(4924.99/(tc+237.1)))/((tc+105.)**1.57)
    esice = np.exp(43.494-(6545.8/(tc+278.)))/((tc+868.)**2.)
    es = xr.where(tc>0.,eswat,esice)/100.
    return es

def calc_qs(p,t):
    '''
    Purpose: Calculate saturation specific humidity (qₛ) using Eq. 4 from Miller SFK. 2018. Atmos. Humidity Eq. 
    Plymouth State Wea. Ctr.
    Args:
    - p (xarray.DataArray): pressure DataArray (hPa)
    - t (xarray.DataArray): temperature DataArray (K)
    Returns:
    - xarray.DataArray: qₛ DataArray (kg/kg)
    '''
    rv = 461.50   
    rd = 287.04   
    epsilon = rd/rv
    es = calc_es(t) 
    qs = (epsilon*es)/(p-es*(1.-epsilon))
    return qs

def calc_thetae(p,t,q=None,ps=None):
    '''
    Purpose: Calculate equivalent potential temperature (θₑ) using Eqs. 43 and 55 from Bolton D. 1980. Mon. Wea. Rev.
    Options to calculate θₑ at the surface, or the saturated θₑ, are given.        
    Args:
    - p (xarray.DataArray): pressure DataArray (hPa)
    - t (xarray.DataArray): temperature DataArray (K)
    - q (xarray.DataArray, optional): specific humidity DataArray (kg/kg); if None, saturated θₑ will be calculated
    - ps (xarray.DataArray, optional): surface pressure DataArray (hPa); if given, θₑ at the surface will be calculated
      (values > 1,000 hPa are clamped to 1,000 hPa to avoid interpolation errors)
    Returns:
    - xarray.DataArray: (regular, surface, or saturated) θₑ DataArray (K)
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

# def filter_above_surface(da,ps):
#     '''
#     Purpose: Remove (set to NaN) any data in an xarray.DataArray that corresponds to pressure levels below the surface.
#     Args:
#     - da (xarray.DataArray): input DataArray
#     - ps (xarray.DataArray): surface pressure DataArray (hPa)
#     Returns:
#     - xarray.DataArray: filtered DataArray
#     '''
#     da = xr.where(da.lev<=ps,da,np.nan)
#     return da

def get_level_above(ptarget,levels,side):
    '''
    Purpose: Find the pressure level immediately above a target pressure. With ascending pressure levels, 'above' means 
    lower pressure/higher altitude.
    Args:
    - ptarget (xarray.DataArray or numpy.ndarray): target pressures
    - levels (numpy.ndarray): 1D array of ascending pressure levels
    - side (str): 'left' or 'right' tie-breaking for numpy.searchsorted
    Returns:
    - numpy.ndarray: array of pressure levels immediately above each target pressure (same shape as 'ptarget')
    '''
    searchidx = np.searchsorted(levels,ptarget,side=side)
    levabove  = levels[np.maximum(searchidx-1,0)]
    return levabove

def get_level_below(ptarget,levels,side):
    '''
    Purpose: Find the pressure level immediately below a target pressure. With ascending pressure levels, 'below' means 
    higher pressure/lower altitude.
    Args:
    - ptarget (xarray.DataArray or numpy.ndarray): target pressures
    - levels (numpy.ndarray): 1D array of ascending pressure levels
    - side (str): 'left' or 'right' tie-breaking for numpy.searchsorted
    Returns:
    - numpy.ndarray: array of pressure levels immediately below each target pressure (same shape as 'ptarget')
    '''
    searchidx = np.searchsorted(levels,ptarget,side=side)
    levbelow  = levels[np.minimum(searchidx,len(levels)-1)]
    return levbelow

def calc_layer_average(da,a,b):
    '''
    Purpose: Calculate pressure-weighted vertical average of an xarray.DataArray between two levels ('a' and 'b'). 
    Expects 'a' > 'b' since we integrate from higher to lower pressure.
    Args:
    - da (xarray.DataArray): input DataArray with 'lev' dimension
    - a (xarray.DataArray): DataArray of bottom boundary pressures (higher values, lower altitude)
    - b (xarray.DataArray): DataArray of top boundary pressures (lower values, higher altitude)
    Returns:
    - xarray.DataArray: layer-averaged DataArray
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
    Purpose: Calculate weights for the boundary layer (PBL) and lower free troposphere (LFT) using Eqs. 5a and 5b from 
    Adames AF, Ahmed F, and Neelin JD. 2021. J. Atmos. Sci.
    Args:
    - ps (xarray.DataArray): surface pressure DataArray (hPa)
    - pbltop (xarray.DataArray): DataArray of pressures at the top of the PBL (hPa)
    - lfttop (xarray.DataArray): DataArray of pressures at the top of the LFT (hPa)
    Returns:
    - tuple[xarray.DataArray,xarray.DataArray]: PBL and LFT weights
    '''
    pblthickness = ps-pbltop
    lftthickness = pbltop-lfttop
    wb = (pblthickness/lftthickness)*np.log((pblthickness+lftthickness)/pblthickness)
    wl = 1-wb
    return wb,wl

def calc_bl_terms(thetaeb,thetael,thetaels,wb,wl):
    '''
    Purpose: Calculate CAPEL, SUBSATL, and BL following Eq. 1 from Ahmed F and Neelin JD. 2021. Geophys. Res. Lett.
    Args:
    - thetaeb (xarray.DataArray): DataArray of θₑ averaged over the PBL (K)
    - thetael (xarray.DataArray): DataArray of θₑ averaged over the LFT (K)
    - thetaels (xarray.DataArray): DataArray of saturated θₑ averaged over the LFT (K)
    - wb (xarray.DataArray): DataArray of PBL weights
    - wl (xarray.DataArray): DataArray of LFT weights
    Returns:
    - tuple[xarray.DataArray,xarray.DataArray,xarray.DataArray]: CAPEL, SUBSATL, and BL DataArrays
    '''
    g       = 9.81
    kappal  = 3.
    thetae0 = 340.
    cape    = ((thetaeb-thetaels)/thetaels)*thetae0
    subsat  = ((thetaels-thetael)/thetaels)*thetae0
    bl      = (g/(kappal*thetae0))*((wb*cape)-(wl*subsat))
    return cape,subsat,bl

def dataset(da,shortname,longname,units,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Wrap a standardized xarray.DataArray into an xarray.Dataset, preserving coordinates and setting variable 
    and global metadata.
    Args:
    - da (xarray.DataArray): input DataArray
    - shortname (str): variable name (abbreviation)
    - longname (str): variable long name/description
    - units (str): variable units
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)    
    Returns:
    - xarray.Dataset: Dataset containing the variable named 'shortname' and metadata
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
    Purpose: Save an xarray.Dataset to a NetCDF file in the specified directory, then verify the write by reopening.
    Args:
    - ds (xarray.Dataset): Dataset to save
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''  
    shortname = list(ds.data_vars)[0]
    #######################################
    filename  = f'{shortname}_unfiltered.nc'
    #######################################
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

# if __name__=='__main__':
#     try:
#         logger.info('Importing all raw variables...')        
#         pr = import_data('IMERG_V06_precipitation_rate.nc')
#         ps = import_data('ERA5_surface_pressure.nc')
#         t  = import_data('ERA5_air_temperature.nc')
#         q  = import_data('ERA5_specific_humidity.nc')
#         logger.info('Resampling/regridding precipitation...')
#         resampledpr = regrid_and_resample(pr.load(),ps.load())
#         del pr
#         logger.info('Beginning chunking...')
#         nyears     = len(np.unique(ps.time.dt.year.values))
#         timechunks = np.array_split(np.arange(len(ps.time)),nyears)
#         results    = {
#             't':[],
#             'q':[],
#             'capeprofile':[],
#             'subsatprofile':[],
#             'cape':[],
#             'subsat':[],
#             'bl':[]}
#         for i,timechunk in enumerate(timechunks):
#             logger.info(f'Processing time chunk {i+1}/{len(timechunks)}...')
#             tchunk  = t.isel(time=timechunk).load()
#             qchunk  = q.isel(time=timechunk).load()
#             pschunk = ps.isel(time=timechunk).load()
#             pchunk  = get_p_array(qchunk)
#             logger.info('   Calculating equivalent potential temperature terms')
#             thetaechunk     = calc_thetae(pchunk,tchunk,qchunk)
#             thetaeschunk    = calc_thetae(pchunk,tchunk)
#             thetaesurfchunk = calc_thetae(pchunk,tchunk,qchunk,pschunk)
#             del pchunk
#             logger.info('   Filtering out data where the pressure level is below the surface')    
#             filteredtchunk = filter_above_surface(tchunk,pschunk)
#             filteredqchunk = filter_above_surface(qchunk,pschunk)
#             filteredthetaechunk  = filter_above_surface(thetaechunk,pschunk)
#             filteredthetaeschunk = filter_above_surface(thetaeschunk,pschunk)
#             del tchunk,qchunk           
#             logger.info('   Calculating CAPE-like and SUBSAT-like profiles')    
#             capeprofilechunk   = thetaesurfchunk-filteredthetaeschunk
#             subsatprofilechunk = filteredthetaeschunk-filteredthetaechunk
#             del thetaesurfchunk,filteredthetaechunk,filteredthetaeschunk
#             logger.info('   Calculating layer averages')
#             pbltopchunk = pschunk-100.
#             lfttopchunk = xr.full_like(pschunk,500.)
#             thetaebchunk    = calc_layer_average(thetaechunk,pschunk,pbltopchunk)*np.sqrt(-1+2*(pschunk>lfttopchunk))
#             thetaelchunk    = calc_layer_average(thetaechunk,pbltopchunk,lfttopchunk)
#             thetaelschunk   = calc_layer_average(thetaeschunk,pbltopchunk,lfttopchunk)
#             wbchunk,wlchunk = calc_weights(pschunk,pbltopchunk,lfttopchunk)
#             del pschunk,pbltopchunk,lfttopchunk,thetaechunk,thetaeschunk
#             logger.info('   Calculating BL terms')
#             capechunk,subsatchunk,blchunk = calc_bl_terms(thetaebchunk,thetaelchunk,thetaelschunk,wbchunk,wlchunk)
#             del wbchunk,wlchunk,thetaebchunk,thetaelchunk,thetaelschunk
#             logger.info('   Appending chunk results')
#             results['t'].append(filteredtchunk)
#             results['q'].append(filteredqchunk)
#             results['capeprofile'].append(capeprofilechunk)
#             results['subsatprofile'].append(subsatprofilechunk)
#             results['cape'].append(capechunk)
#             results['subsat'].append(subsatchunk)
#             results['bl'].append(blchunk)
#             del filteredtchunk,filteredqchunk,capeprofilechunk,subsatprofilechunk,capechunk,subsatchunk,blchunk
#         del ps,t,q
#         logger.info('Concatenating results and saving...')
#         dslist = [
#             dataset(resampledpr,'pr','Resampled/regridded precipitation rate','mm/day'),
#             dataset(xr.concat(results['t'],dim='time'),'t','Filtered air temperature','K'),
#             dataset(xr.concat(results['q'],dim='time'),'q','Filtered specific humidity','kg/kg'),
#             dataset(xr.concat(results['capeprofile'],dim='time'),'capeprofile','θₑ(surface) - saturated θₑ(p)','K'),
#             dataset(xr.concat(results['subsatprofile'],dim='time'),'subsatprofile','Saturated θₑ(p) - θₑ(p)','K'),   
#             dataset(xr.concat(results['cape'],dim='time'),'cape','Undilute buoyancy in the lower troposphere','K'),
#             dataset(xr.concat(results['subsat'],dim='time'),'subsat','Lower free-tropospheric subsaturation','K'),
#             dataset(xr.concat(results['bl'],dim='time'),'bl','Average buoyancy in the lower troposphere','m/s²')]
#         for ds in dslist:
#             save(ds)
#             del ds
#         logger.info('Script execution completed successfully!')
#     except Exception as e:
#         logger.error(f'An unexpected error occurred: {str(e)}')

if __name__=='__main__':
    try:
        logger.info('Importing all raw variables...')        
        pr = import_data('IMERG_V06_precipitation_rate.nc')
        ps = import_data('ERA5_surface_pressure.nc')
        t  = import_data('ERA5_air_temperature.nc')
        q  = import_data('ERA5_specific_humidity.nc')
        logger.info('Resampling/regridding precipitation...')
        resampledpr = regrid_and_resample(pr.load(),ps.load())
        del pr
        logger.info('Beginning chunking...')
        nyears     = len(np.unique(ps.time.dt.year.values))
        timechunks = np.array_split(np.arange(len(ps.time)),nyears)
        results    = {
            't':[],
            'q':[],
            'capeprofile':[],
            'subsatprofile':[],
            'cape':[],
            'subsat':[],
            'bl':[]}
        for i,timechunk in enumerate(timechunks):
            logger.info(f'Processing time chunk {i+1}/{len(timechunks)}...')
            tchunk  = t.isel(time=timechunk).load()
            qchunk  = q.isel(time=timechunk).load()
            pschunk = ps.isel(time=timechunk).load()
            pchunk  = get_p_array(qchunk)
            logger.info('   Calculating equivalent potential temperature terms')
            thetaechunk     = calc_thetae(pchunk,tchunk,qchunk)
            thetaeschunk    = calc_thetae(pchunk,tchunk)
            thetaesurfchunk = calc_thetae(pchunk,tchunk,qchunk,pschunk)
            del pchunk        
            logger.info('   Calculating CAPE-like and SUBSAT-like profiles')    
            capeprofilechunk   = thetaesurfchunk-thetaeschunk
            subsatprofilechunk = thetaeschunk-thetaechunk
            del thetaesurfchunk
            logger.info('   Calculating layer averages')
            pbltopchunk = pschunk-100.
            lfttopchunk = xr.full_like(pschunk,500.)
            thetaebchunk    = calc_layer_average(thetaechunk,pschunk,pbltopchunk)*np.sqrt(-1+2*(pschunk>lfttopchunk))
            thetaelchunk    = calc_layer_average(thetaechunk,pbltopchunk,lfttopchunk)
            thetaelschunk   = calc_layer_average(thetaeschunk,pbltopchunk,lfttopchunk)
            wbchunk,wlchunk = calc_weights(pschunk,pbltopchunk,lfttopchunk)
            del pschunk,pbltopchunk,lfttopchunk,thetaechunk,thetaeschunk
            logger.info('   Calculating BL terms')
            capechunk,subsatchunk,blchunk = calc_bl_terms(thetaebchunk,thetaelchunk,thetaelschunk,wbchunk,wlchunk)
            del wbchunk,wlchunk,thetaebchunk,thetaelchunk,thetaelschunk
            logger.info('   Appending chunk results')
            results['t'].append(tchunk)
            results['q'].append(qchunk)
            results['capeprofile'].append(capeprofilechunk)
            results['subsatprofile'].append(subsatprofilechunk)
            results['cape'].append(capechunk)
            results['subsat'].append(subsatchunk)
            results['bl'].append(blchunk)
            del capeprofilechunk,subsatprofilechunk,capechunk,subsatchunk,blchunk,tchunk,qchunk
        del ps,t,q
        logger.info('Concatenating results and saving...')
        dslist = [
            dataset(resampledpr,'pr','Resampled/regridded precipitation rate','mm/day'),
            dataset(xr.concat(results['t'],dim='time'),'t','Air temperature','K'),
            dataset(xr.concat(results['q'],dim='time'),'q','Specific humidity','kg/kg'),
            dataset(xr.concat(results['capeprofile'],dim='time'),'capeprofile','θₑ(surface) - saturated θₑ(p)','K'),
            dataset(xr.concat(results['subsatprofile'],dim='time'),'subsatprofile','Saturated θₑ(p) - θₑ(p)','K'),   
            dataset(xr.concat(results['cape'],dim='time'),'cape','Undilute buoyancy in the lower troposphere','K'),
            dataset(xr.concat(results['subsat'],dim='time'),'subsat','Lower free-tropospheric subsaturation','K'),
            dataset(xr.concat(results['bl'],dim='time'),'bl','Average buoyancy in the lower troposphere','m/s²')]
        for ds in dslist:
            save(ds)
            del ds
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')