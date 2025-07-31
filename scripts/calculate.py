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
CHUNKSIZE = 21

def get_data(filename,filedir=FILEDIR):
    '''
    Purpose: Open a NetCDF file as an xarray.DataArray and check level ordering (if applicable).
    Args:
    - filename (str): name of the file to load
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - xarray.DataArray: loaded DataArray (with level coordinate ordered if applicable) 
    '''
    filepath = os.path.join(filedir,filename)
    da  = xr.open_dataarray(filepath)
    if 'lev' in da.dims:
        if not da.lev.diff('lev').all()>0:
            da = da.reindex(lev=sorted(da.lev))
            logger.info(f'Levels reordered from {da.lev.values[0]} to {da.lev.values[-1]} hPa')
    return da
    
def get_p_array(da):
    '''
    Purpose: Create a pressure xarray.DataArray from the 'lev' coordinate.
    Args:
    - da (xarray.DataArray): input DataArray containing the 'lev' coordinate
    Returns:
    - xarray.DataArray: pressure DataArray
    '''
    p = da.lev.expand_dims({'time':da.time,'lat':da.lat,'lon':da.lon}).transpose('lev','time','lat','lon')
    return p

def regrid_and_resample(da,gridtarget,frequency='H',method='first'):
    '''
    Purpose: Regrid and resample a DataArray.
    Args:
    - da (xarray.DataArray): input DataArray
    - gridtarget (xarray.DataArray): DataArray that contains the target grid for regridding
    - frequency (str): resampling frequency (defaults to 'H' for hourly)
    - method (str): resampling method (defaults to 'first')
    Returns:
    - xarray.DataArray: regridded and resampled DataArray
    '''
    regridder = xesmf.Regridder(da,gridtarget,method='bilinear')
    da = regridder(da,keep_attrs=True)
    da.coords['time'] = da.time.dt.floor(frequency) 
    da = da.groupby('time').mean()
    return da
    
def calc_es(t):
    '''
    Purpose: Calculate saturation vapor pressure (eₛ) following Eqs. 17 and 18 from: Huang J. 2018. J. Appl. Meteorol. Climatol.
    Args:
    - t (xarray.DataArray): temperature DataArray (K) 
    Returns:
    - xarray.DataArray: saturation vapor pressure DataArray (hPa)
    '''    
    tc  = t-273.15
    esw = np.exp(34.494-(4924.99/(tc+237.1)))/((tc+105.)**1.57)
    esi = np.exp(43.494-(6545.8/(tc+278.)))/((tc+868.)**2.)
    es  = np.where(tc>0.,esw,esi) 
    es  = es/100.
    return es

def calc_qs(p,t):
    '''
    Purpose: Calculate saturation specific humidity (qₛ) following Eq. 4 from: Miller SFK. 2018. Atmos. Humidity Eq. Plymouth State Wea. Ctr.
    Args:
    - p (xarray.DataArray): pressure DataArray (hPa)
    - t (xarray.DataArray): temperature DataArray (K)
    Returns:
    - xarray.DataArray: saturation specific humidity DataArray (kg/kg)
    '''
    rv = 461.50   
    rd = 287.04    
    es = calc_es(t) 
    epsilon = rd/rv
    qs = (epsilon*es)/(p-es*(1.-epsilon))
    return qs

def calc_thetae(p,t,q=None,ps=None):
    '''
    Purpose: Calculate equivalent potential temperature (θe) following Eqs. 43 and 55 from Bolton D. 1980. Mon. Wea. Rev. Options to calculate θe at the surface, or the saturated θe are given.        
    Args:
    - p (xarray.DataArray): pressure DataArray (hPa)
    - t (xarray.DataArray): temperature DataArray (K)
    - q (xarray.DataArray, optional): specific humidity DataArray (kg/kg); if None, saturated θe will be calculated
    - ps (xarray.DataArray, optional): surface pressure DataArray (hPa); if given, θe at the surface will be calculated
    Returns:
    - xarray.DataArray: (regular, surface, or saturated) θe DataArray (K)
    '''
    if q is None:
        q = calc_qs(p,t)
    if ps is not None:
        t = t.interp(lev=ps)
        q = q.interp(lev=ps)
        p = ps
        pvalues = ps
    else:
        pvalues = p.lev
    p0 = 1000.  
    rv = 461.5  
    rd = 287.04
    epsilon = rd/rv
    r  = q/(1.-q) 
    e  = (pvalues*r)/(epsilon+r)
    tl = 2840./(3.5*np.log(t)-np.log(e)-4.805)+55.
    thetae = t*(p0/p)**(0.2854*(1.-0.28*r))*np.exp((3.376/tl-0.00254)*1000.*r*(1.+0.81*r))
    return thetae

def filter_above_surface(da,ps):
    '''
    Purpose: Remove (set to NaN) any data in an xarray.DataArray that corresponds to pressure levels below the surface.
    Args:
    - da (xarray.DataArray): input DataArray
    - ps (xarray.DataArray): surface pressure DataArray (hPa)
    Returns:
    - xarray.DataArray: filtered DataArray
    '''
    da = xr.where(da.lev<=ps,da,np.nan)
    return da

def get_adjacent_pressure(pref,levs,side,direction):
    '''
    Purpose: Find the pressure level immediately above or below a given pressure.
    Args:
    - pref (xr.DataArray): DataArray of reference pressures (hPa)
    - levs (np.ndarray): 1D Array of pressure levels (hPa)
    - side (str): 'left' or 'right' for numpy.searchsorted
    - direction (str): 'above' or 'below' to specify which adjacent level to find
    Returns:
    - xr.DataArray: DataArray of pressure levels immediately above or below 'pref'
    '''
    if direction=='above':
        return xr.apply_ufunc(lambda x:levs[np.maximum(np.searchsorted(levs,x,side=side)-1,0)],pref)
    elif direction=='below':
        return xr.apply_ufunc(lambda x:levs[np.minimum(np.searchsorted(levs,x,side=side),len(levs)-1)],pref)
    else:
        raise ValueError("Direction must be either 'above' or 'below'")

def integrate_partial_layer(pref,pabove,pbelow,daabove,dabelow,bound):
    '''
    Purpose: Integrate over a partial layer, including correction for edge cases.
    Args:
    - pref (xr.DataArray): DataArray of reference pressures for integration (hPa)
    - pabove (xr.DataArray): DataArray of pressure levels above 'pref' (hPa)
    - pbelow (xr.DataArray): DataArray of pressure levels above 'pref' (hPa)
    - daabove (xr.DataArray): DataArray of values to integrate at 'pabove'
    - dabelow (xr.DataArray): DataArray of values to integrate at 'pbelow'
    - bound (bool): 'lower' or 'upper' to specify which correction to apply
    Returns:
    - xr.DataArray: DataArray of values integrated over the partial layer
    '''
    dp = pbelow-pabove
    dp = xr.where(dp==0,1e-8,dp)
    if bound=='upper':
        correction = -daabove/2*dp*(pref<pabove.min())
        integral   = (pref-pabove)*daabove+(dabelow-daabove)*(pref-pabove)**2/(2*dp)+correction
        return integral
    elif bound=='lower':
        correction = -dabelow/2*dp*(pref>pbelow.max())
        integral   = (pbelow-pref)*daabove+(dabelow-daabove)*dp*(1-((pref-pabove)/dp)**2)/2+correction
        return integral
    else:
        raise ValueError("Bound must be either 'upper' or 'lower'")

def calc_layer_average(da,upperbound,lowerbound):
    '''
    Purpose: Calculate vertical average of an xarray.DataArray between two pressure levels.
    Args:
    - da (xr.DataArray): input DataArray with 'lev' coordinate
    - upperbound (xr.DataArray): DataArray of upper pressure level bound (hPa)
    - lowerbound (xr.DataArray): DataArray of lower pressure level bound (hPa)
    Returns:
    - xr.DataArray: layer-averaged DataArray
    '''
    da,upperbound,lowerbound = da.load(),upperbound.load(),lowerbound.load()
    paboveupper   = get_adjacent_pressure(upperbound,np.array(da.lev),'right','above')
    pbelowupper   = get_adjacent_pressure(upperbound,np.array(da.lev),'right','below')
    upperintegral = integrate_partial_layer(upperbound,paboveupper,pbelowupper,da.sel(lev=paboveupper),da.sel(lev=pbelowupper),bound='upper')
    pabovelower   = get_adjacent_pressure(lowerbound,np.array(da.lev),'left','above')
    pbelowlower   = get_adjacent_pressure(lowerbound,np.array(da.lev),'left','below')
    lowerintegral = integrate_partial_layer(lowerbound,pabovelower,pbelowlower,da.sel(lev=pabovelower),da.sel(lev=pbelowlower),bound='lower')
    innerintegral = (da*(da.lev<=upperbound)*(da.lev>=lowerbound)).integrate('lev')
    layeraverage  = (upperintegral+innerintegral-lowerintegral)/(upperbound-lowerbound)
    return layeraverage
    
def calc_weights(ps,pbltop,lfttop):
    '''
    Purpose: Calculate weights for the boundary layer (PBL) and lower free troposphere (LFT) following Adames AF et al. 2021. J. Atmos. Sci.
    Args:
    - ps (xr.DataArray): surface pressure DataArray (hPa)
    - pbltop (xr.DataArray): DataArray of pressures at the top of the PBL (hPa)
    - lfttop (xr.DataArray): DataArray of pressures at the top of the LFT (hPa)
    Returns:
    - tuple (xr.DataArray, xr.DataArray): containing DataArrays of PBL and LFT weights
    '''
    pblthickness = ps-pbltop
    lftthickness = pbltop-lfttop
    wb = (pblthickness/lftthickness)*np.log((pblthickness+lftthickness)/pblthickness)
    wl = 1-wb
    return wb,wl

def calc_bl_terms(thetaeb,thetael,thetaels,wb,wl):
    '''
    Purpose: Calculate BL, CAPEL, sand SUBSATL following Eq. 1 from: Ahmed F et al. 2021. Geophys. Res. Lett.
    Args:
    - thetaeb (xr.DataArray): DataArray of equivalent potential temperature averaged over the PBL
    - thetael (xr.DataArray): DataArray of equivalent potential temperature averaged over the LFT
    - thetaels (xr.DataArray): DataArray of saturated equivalent potential temperature averaged over the LFT
    - wb (xr.DataArray): DataArray of weights for the PBL
    - wl (xr.DataArray): DataArray of weights for the LFT
    Returns:
    - tuple (xr.DataArray, xr.DataArray): containing DataArrays of CAPEL, SUBSATL, and BL
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
    Purpose: Format an xarray.DataArray into an xarray.Dataset with standard attributes and metadata.
    Args:
    - da (xarray.DataArray): input DataArray
    - shortname (str): variable name abbreviation
    - longname (str): full variable name
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

def save(ds,savedir=SAVEDIR):
    '''
    Purpose: Save an xarray.Dataset to a NetCDF file in the specified directory. Verify the file was saved successfully by attempting to reopen it.
    Args:
    - ds (xarray.Dataset): Dataset to save
    - savedir (str): directory where the file should be saved (defaults to SAVEDIR)
    Returns:
    - bool: True if the save operation was successful, False otherwise
    '''  
    shortname = list(ds.data_vars)[0]
    filename  = f'{shortname}.nc'
    filepath  = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filename}...')   
    try:
        ds.to_netcdf(filepath)
        logger.info(f'File writing successful: {filename}')
        with xr.open_dataset(filepath) as test:
            pass
        logger.info(f'File verification successful: {filename}')
        return True
    except Exception as e:
        logger.error(f'Failed to save or verify {filename}: {e}')
        return False

if __name__=='__main__':
    try:
        logger.info('Importing all raw variables...')        
        pr = get_data('IMERG_V06_precipitation_rate.nc')
        ps = get_data('ERA5_surface_pressure.nc')
        t  = get_data('ERA5_air_temperature.nc')
        q  = get_data('ERA5_specific_humidity.nc')
        logger.info('Resampling/regridding precipitation...')
        resampledpr = regrid_and_resample(pr.load(),ps.load())
        del pr
        logger.info('Beginning chunking...')
        timechunks   = np.array_split(np.arange(len(ps.time)),CHUNKSIZE)
        chunkresults = {
            't':[],
            'q':[],
            'capeprofile':[],
            'subsatprofile':[],
            'cape':[],
            'subsat':[],
            'bl':[]}
        for i,timechunk in enumerate(timechunks):
            logger.info(f'Processing time chunk {i+1}/{len(timechunks)}...')
            logger.info('   Loading in raw ERA5 variables chunks...')
            tchunk  = t.isel(time=timechunk).load()
            qchunk  = q.isel(time=timechunk).load()
            pschunk = ps.isel(time=timechunk).load()
            pchunk  = get_p_array(qchunk)
            logger.info('   Calculate equivalent potential temperature terms...')
            thetaechunk     = calc_thetae(pchunk,tchunk,qchunk)
            thetaeschunk    = calc_thetae(pchunk,tchunk)
            thetaesurfchunk = calc_thetae(pchunk,tchunk,qchunk,pschunk)
            del pchunk
            logger.info('   Filter out data where the pressure level is below the surface...')    
            filteredtchunk = filter_above_surface(tchunk,pschunk)
            filteredqchunk = filter_above_surface(qchunk,pschunk)
            filteredthetaechunk  = filter_above_surface(thetaechunk,pschunk)
            filteredthetaeschunk = filter_above_surface(thetaeschunk,pschunk)
            del tchunk,qchunk,thetaechunk,thetaeschunk
            logger.info('  Calculate CAPE-like and SUBSAT-like profiles...')    
            capeprofilechunk   = thetaesurfchunk-filteredthetaeschunk
            subsatprofilechunk = filteredthetaeschunk-filteredthetaechunk
            del thetaesurfchunk
            logger.info('   Calculating BL terms...')
            pbltopchunk = pschunk-100.
            lfttopchunk = xr.full_like(pschunk,500.) 
            thetaebchunk    = calc_layer_average(filteredthetaechunk,pschunk,pbltopchunk)*np.sqrt(-1+2*(pschunk>lfttopchunk))
            thetaelchunk    = calc_layer_average(filteredthetaechunk,pbltopchunk,lfttopchunk)
            thetaelschunk   = calc_layer_average(filteredthetaeschunk,pbltopchunk,lfttopchunk)
            wbchunk,wlchunk = calc_weights(pschunk,pbltopchunk,lfttopchunk)
            capechunk,subsatchunk,blchunk = calc_bl_terms(thetaebchunk,thetaelchunk,thetaelschunk,wbchunk,wlchunk)
            del pbltopchunk,lfttopchunk,wbchunk,wlchunk
            logger.info('   Appending chunk results...')
            chunkresults['t'].append(filteredtchunk)
            chunkresults['q'].append(filteredqchunk)
            chunkresults['capeprofile'].append(capeprofilechunk)
            chunkresults['subsatprofile'].append(subsatprofilechunk)
            chunkresults['cape'].append(capechunk)
            chunkresults['subsat'].append(subsatchunk)
            chunkresults['bl'].append(blchunk)
            del (pschunk,filteredtchunk,filteredqchunk,filteredthetaechunk,filteredthetaeschunk,capeprofilechunk,subsatprofilechunk,
                 thetaebchunk,thetaelchunk,thetaelschunk,capechunk,subsatchunk,blchunk)
        del ps,t,q
        logger.info('Concatenating results and saving...')
        finalresults = {}
        for key,chunks in chunkresults.items():
            finalresults[key] = xr.concat(chunks,dim='time')
            del chunks
        dslist = [
            dataset(resampledpr,'pr','Resampled/regridded precipitation rate','mm/day'),
            dataset(finalresults['t'],'t','Filtered air temperature','K'),
            dataset(finalresults['q'],'q','Filtered specific humidity','kg/kg'),
            dataset(finalresults['capeprofile'],'capeprofile','Equivalent potential temperature at the surface minus saturated equivalent potential temperature','K'),
            dataset(finalresults['subsatprofile'],'subsatprofile','Saturated equivalent potential temperature minus equivalent potential temperature','K'),   
            dataset(finalresults['cape'],'cape','Undilute buoyancy in the lower troposphere','K'),
            dataset(finalresults['subsat'],'subsat','Lower free-tropospheric subsaturation','K'),
            dataset(finalresults['bl'],'bl','Average buoyancy in the lower troposphere','m/s²')]
        for ds in dslist:
            save(ds)
            del ds
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')