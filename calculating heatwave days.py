import xarray as xr
import matplotlib.pyplot as plt
from cartopy import feature, crs
import numpy as np
import pymannkendall as mk
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import SubplotSpec
warnings.filterwarnings('ignore')

path='/media/kenz/1B8D1A637BBA134B/CHIRTS'

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    row.set_title(f'{title}\n', fontweight='semibold')#) if row==1 else row.set_title(f'{title}')
    # if row==0 row.set_title(f'{title}\n\n\n', fontweight='semibold') else row.set_title(f'{title}\n'
    row.set_frame_on(False)
    row.axis('off')

def df_read(path,file,var):
    ds = xr.open_dataset(f'{path}/{file}')[f'{var}'].sortby('time').drop_duplicates('time').sel(time=slice('1983',None))
    return ds
def sens_slopes(data):
    data = data.sel(longitude=np.arange(-19.875, 21.875,0.25), latitude=np.arange(3.125,26.875,0.25), method = 'nearest')
    output=[]
    for i in np.arange(len(data.latitude.values)):
        for j in np.arange(len(data.longitude.values)):
            try:
                slope_val = mk.sens_slope(data[:,i,j]).slope
            except:
                slope_val = np.nan
            output.append(slope_val)
    output = np.copy(output).reshape(data.latitude.size,data.longitude.size)
    slopes=xr.DataArray(output, dims=('latitude','longitude'), coords={'latitude':data.latitude,'longitude':data.longitude})
    return slopes 

def CTpctl(path,temp,pctl,var):
    temp = df_read(path,temp,var)
    mask = temp.groupby('time.month') > pctl
    return temp.where(mask, drop=True)

def rising_filter(array, axis):
    assert(array.shape[axis] == 5)
    assert(axis == array.ndim-1 or axis == -1)
    left = array[..., 1]
    right = array[..., 2:].sum(axis=axis)
    return np.logical_and(np.isnan(left), np.isfinite(right))
def rising_filter_dask(x, dim):
    return xr.apply_ufunc(rising_filter, x, input_core_dims=[[dim]],
                             kwargs={'axis': -1},
                             dask='parallelized',
                             output_dtypes=[bool])
def HWD(data):
    windows = data.chunk({'time':20}).rolling(time=5, center=True, min_periods=1).construct('rolling_dim')
    heatwave_starts = rising_filter_dask(windows, dim='rolling_dim')
    return heatwave_starts
def set_fig_params(ax):
    for i,j in enumerate(ax):
    # for i in (range(0,len(axes))):
        ax[i].set_extent([-19,15,4,24])
        ax[i].add_feature(feature.COASTLINE)
        ax[i].add_feature(feature.BORDERS)
        ax[i].add_feature(feature.STATES, linewidth = 0.2)
        ax[i].set_xticks([-20,-10,0,10], crs=crs.PlateCarree())
        ax[i].set_yticks([5,10,15,20], crs=crs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax[i].xaxis.set_major_formatter(lon_formatter)
        ax[i].yaxis.set_major_formatter(lat_formatter)
        
tmean_ehf = 'Tmean/ehf.nc'
tmax = 'Tmax/chirts.Tmax.1983.2016.WA.days_p25.nc'
tmax90 = 'Tmax/chirts.Tmax.1983.2016.WA.days_p25_ymonpctl.nc'
tmin90 = 'Tmin/chirts.Tmin.1983.2016.WA.days_p25_ymonpctl.nc'
tmin = 'Tmin/chirts.Tmin.1983.2016.WA.days_p25.nc'

tmx = 'Tmax/tx.nc'
tmn = 'Tmin/tn.nc'

tmean = 'Tmean/chirts.Tmean.1983.2016.WA.days_p25.nc'

tx90 = df_read(path,tmax90,'Tmax')
tx90 = tx90.groupby('time.month').mean('time')

tn90 = df_read(path,tmin90,'Tmin')
tn90 = tn90.groupby('time.month').mean('time')

txx = df_read(path, tmax, 'Tmax')
tnn = df_read(path, tmin, 'Tmin')

tmean_d = df_read(path,tmean,'Tmax')

tx = df_read(path, tmx, 'Tmax')
tn = df_read(path, tmn, 'Tmin')
ehf = df_read(path,tmean_ehf,'Tmax')
ds_gtehf = ehf.groupby('time.year').count()

##### ctx90 for rolling percentile
# %%time
windows = tx.chunk({'time':20}).rolling(time=5, center=True, min_periods=1).construct('rolling_dim')
hwn_tx_roll = rising_filter_dask(windows, dim='rolling_dim')
windows = tn.chunk({'time':20}).rolling(time=5, center=True, min_periods=1).construct('rolling_dim')
hwn_tn_roll = rising_filter_dask(windows, dim='rolling_dim')

#### Using rolling percentile ####
# %%time

# Create subplots
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})
ax = ax.flatten()
set_fig_params(ax)

vmax=15
cc='YlOrBr'

hwdtx = hwn_tx_roll.groupby('time.year').sum('time').max('year')
cb1 = hwdtx.where(hwdtx>0).plot(ax=ax[0], cmap=cc, vmax=vmax, add_colorbar=False)
ax[0].set_title('CTx90 1983-2016')

### ds_gtn.mean('year').plot(ax=ax[1], cmap=cc, vmax=60, add_colorbar=False)
# ds_gtn.where(ds_gtn>0).mean('year').plot(ax=ax[1], cmap=cc, vmax=55,  vmin=35, add_colorbar=False)

hwdtn = hwn_tn_roll.groupby('time.year').sum('time').max('year')
hwdtn.where(hwdtn>0).plot(ax=ax[1], cmap=cc, vmax=vmax, add_colorbar=False)
ax[1].set_title('CTn90 1983-2016')

ds_gtehf.where(ds_gtehf>0).mean('year').plot(ax=ax[2], cmap=cc, vmax=vmax, add_colorbar=False)
ax[2].set_title('EHF 1983-2016')

cm = 'RdBu_r'
vm = 1

txsl = hwn_tx_roll.groupby('time.year').sum('time')
cb2 = sens_slopes(txsl.where(txsl>0).load()).plot(ax=ax[3], vmax=vm, cmap=cm, add_colorbar=False)

tnsl = hwn_tn_roll.groupby('time.year').sum('time')
sens_slopes(tnsl.where(tnsl>0).load()).plot(ax=ax[4], vmax=vm, cmap=cm, add_colorbar=False)

sens_slopes(ds_gtehf.where(ds_gtehf>0).load()).plot(ax=ax[5], cmap=cm, vmax=vm, add_colorbar=False)
# ax[5].set_title('EHF')

label = ['HWD','Trends']

cb = [cb1, cb2]
# Add colorbars
for i, j in enumerate([0.62, 0.13]):
    cax = fig.add_axes([1, j, 0.02, 0.3])
    fig.colorbar(cb[i], cax=cax, orientation='vertical', extend='both', label = label[i])

fig.tight_layout()
plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/hwd_.jpeg', bbox_inches='tight')

#### Rolling pctl
ds = hwn_tx_roll.groupby('time.year').sum('time').idxmax('year').load().astype(int)#.plot(ax = ax,cmap='Set2')

####tx#### 1
plt.figure(figsize=(6,3))
txx.sel(latitude=21, longitude=15, method='nearest').sel(time=slice('2008-10-27', '2008-11-16')).plot(alpha=0.3, label='Temperature')
region = tx.sel(latitude=21, longitude=15, method='nearest').sel(time=slice('2008-10-27', '2008-11-16'))#,'1999'))
region.where(
    region.rolling(time=1).count()>=1
).plot(marker='*', alpha=0.3, label='Heat wave days')
region.where(
    region.rolling(time=10).count()>=10
).plot(marker='.')
plt.grid()
plt.title('')
plt.legend()
plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/hwd_tx_line.eps', bbox_inches='tight')

##### tn
plt.figure(figsize=(6,3))
tnn.sel(latitude=23, longitude=15, method='nearest').sel(time=slice('1988-10-04', '1988-10-24')).plot(alpha=0.3, label='Temperature')
region = tn.sel(latitude=23, longitude=15, method='nearest').sel(time=slice('1988-10-04', '1988-10-24'))
region.where(
    region.rolling(time=1).count()>=1
).plot(marker='*', alpha=0.3, label='Heat wave day')
region.where(
    region.rolling(time=11).count()>=11
).plot(marker='.')
plt.title('')
plt.grid()
plt.legend()
plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/hwd_tn_line.eps', bbox_inches='tight')
##### tn
plt.figure(figsize=(6,3))
tnn.sel(latitude=23, longitude=15, method='nearest').sel(time=slice('1988-10-04', '1988-10-24')).plot(alpha=0.3, label='Temperature')
region = tn.sel(latitude=23, longitude=15, method='nearest').sel(time=slice('1988-10-04', '1988-10-24'))
region.where(
    region.rolling(time=1).count()>=1
).plot(marker='*', alpha=0.3, label='Heat wave day')
region.where(
    region.rolling(time=11).count()>=11
).plot(marker='.')
plt.title('')
plt.grid()
plt.legend()
plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/hwd_tn_line.eps', bbox_inches='tight')

##### ehf #####
plt.figure(figsize=(6,3))
g = ehf.sel(latitude=8.46, longitude=-11.79, method='nearest').sel(time=slice('2011-03-14','2011-04-16'))
tx_r = tmean_d.sel(latitude=8.46, longitude=-11.79, method='nearest').sel(time=slice('2011-03-14','2011-04-16'))

tx_r.plot(label = 'Temperature')
tx_r.sel(time=slice('2011-03-19','2011-04-10')).plot(marker='*', label='Heat wave day')
region = tx_r.where(g>0)
region.where(
    region.rolling(time=10, center=True, min_periods=1).count()>=7
).plot(marker='*', color='orange')

# region.where(
#     region.rolling(time=1).count()>=1
# ).plot(marker='.')
plt.title('')
plt.grid()
plt.legend()
plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/ehf_line.eps', bbox_inches='tight')
hdtn = hwn_tn.groupby('time.year').sum('time').max('year').sel(longitude=slice(-20,15), latitude=slice(0,18))
data_str_tn = -(xr.open_dataset('../Era_5/olr_t2.nc').ttr/3600*3)
data_sst_tn = xr.open_dataset('../Era_5/ss-temp.nc').sst-273.15
# tnn

# level =1000
def plt_map_roll(hd,data,tmp,data_str,data_sst,dates,level,wind_thres,temp_thres, rad_thres, sst_thres,rh_thres):
    import matplotlib as mpl
    def show_axes():
        for i,j in enumerate(ax):
            ax[i].set_extent([-22,25,1,30])
            ax[i].add_feature(feature.COASTLINE)
            ax[i].add_feature(feature.BORDERS)
            ax[i].add_feature(feature.STATES, linewidth = 0.2)
            ax[i].set_xticks([-20,-10,0,10,20], crs=crs.PlateCarree())
            ax[i].set_yticks([10,20,27], crs=crs.PlateCarree())
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax[i].xaxis.set_major_formatter(lon_formatter)
            ax[i].yaxis.set_major_formatter(lat_formatter)
    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(15, 10),sharex=True, subplot_kw={'projection': crs.PlateCarree()})
    ax = ax.flatten()
    show_axes()
    cmap = mpl.cm.rainbow
    cm = 'pink_r'
    # norm = mpl.colors.Normalize(vmax=wind_thres[0], vmin=wind_thres[1])
    # norm = mpl.colors.Normalize(vmin=0, vmax=int(wind_speed.max()))
    norm = mpl.colors.Normalize(vmin=0, vmax=wind_thres)
    # dates = ['1999-07-12', '1999-07-17', '1999-07-22', '1999-07-27']
    days = ['Prior 5 days', 'Day 1', 'Day 5', 'Post 5 days']

    for i in range(0,4):
        hwnp = hd.where(hd>0).plot(ax=ax[i],cbar_kwargs={'orientation': 'horizontal', 'label':'HWD'}, vmax = 10, vmin=2, cmap='bone_r')
        ax[i].set_title(days[i])
    lon = [-30,25]
    lat = [0,35]
    for i, j in enumerate(dates):    
        rhp = data.r.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0])).sortby('latitude').sel(time=f'{dates[i]}T15', method='nearest').plot.contour(
            ax=ax[i+4],add_colorbar=False, cmap= 'rainbow', vmin=rh_thres[1], vmax=rh_thres[0])
        temp = tmp.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[0],lat[1])).sel(time=f'{dates[i]}', method = 'nearest').plot(
            ax=ax[i+4],cmap = cm,cbar_kwargs={'orientation': 'horizontal','label':'Temperature [$^o$C]'},vmax=temp_thres[0], vmin=temp_thres[1])
        radp = data_str.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0])).sortby('latitude').sel(
            time=f'{dates[i]}T15', method = 'nearest').plot(ax=ax[8+i],cbar_kwargs={
            'orientation': 'horizontal', 'label':'OLR [$Wm^2$C]'}, cmap= 'rainbow', vmax = rad_thres[0], vmin=rad_thres[1])
        sstp = data_sst.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0])).sortby('latitude').sel(time=f'{dates[i]}T15').plot(
            ax=ax[8+i],add_colorbar=False, cmap = 'jet', vmax=sst_thres[0], vmin=sst_thres[1])
 
        ds = data.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0])).sortby('latitude').drop(
            ['z','r','t','w','q']).sel(time=f'{dates[i]}T15', method='nearest')
        wind_speed = np.sqrt(ds.u[::3, ::3] ** 2 + ds.v[::3, ::3] ** 2)
        # norm = mpl.colors.Normalize(vmin=0, vmax=int(wind_speed.max()))
        quiver = ax[i].quiver(ds.u[::3,::3].longitude, ds.u[::3,::3].latitude, 
                               ds.u[::3,::3], ds.v[::3,::3],wind_speed, scale=1200-int(level[:-3]),
                              cmap='rainbow',linewidth=1.50, norm=norm, headwidth=5, headlength=5)

    for i in range(0,12):
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(None)

    cbmap = [rhp,sstp]#,strm]
    lb = ['Relative Humidity [%]','Sea Surface Temperature [$^o$C]']
    for i, j in enumerate([0.45, 0.13]):#, 0.18]):
        cax = fig.add_axes([1, j, 0.02, 0.15])
        fig.colorbar(cbmap[i], cax=cax, orientation='vertical', label = lb[i])

    cax = fig.add_axes([1, 0.77, 0.02, 0.15])
    # cbar = fig.colorbar(strm.lines, cax=cax, orientation='vertical', label = 'wind speed [ms$^-2$]')
    cbar = fig.colorbar(quiver, cax=cax, orientation='vertical', label='wind speed [ms$^-2$]')

    fig.suptitle(f'Pressure level = {level[:-3]} hpa', fontweight ='bold')
    plt.tight_layout()
    # plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/hwd_tn_line.jpeg', bbox_inches='tight')
    plt.savefig(f'/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/temp_wind_radiation_rh_hwn_sst_rollin_pctl_tx{level}_{dates[0]}_1983-2016.jpeg', bbox_inches='tight')
    
### rolling
hdtxx = hwn_tx_roll.groupby('time.year').sum('time').max('year')
# hwn
# .sel(latitude=21, longitude=15, method='nearest').sel(time=slice('2008-10-30', '2008-11-13'))

data_ttr = -(xr.open_dataset('../Era_5/olr_t2.nc').ttr/3600*3)
data_sst = xr.open_dataset('../Era_5/sst.nc').sst-273.15

data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=950)
plt_map_roll(hdtxx, data, txx, data_ttr,data_sst,dates = ['2008-10-27', '2008-11-01', '2008-11-06', '2008-11-14'],level = '950 tx',
        wind_thres = 18,temp_thres = [40,20], rad_thres = [1100,350], sst_thres = [33,18], rh_thres=[100,0])
        
        data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=850)
plt_map_roll(hdtxx, data, txx, data_ttr,data_sst,dates = ['2008-10-27', '2008-11-01', '2008-11-06', '2008-11-14'],level = '850 tx',
        wind_thres = 20,temp_thres = [40,20], rad_thres = [1100,350], sst_thres = [33,18], rh_thres=[120,0])
        
        data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=500)
plt_map_roll(hdtxx, data, txx, data_ttr,data_sst,dates = ['2008-10-27', '2008-11-01', '2008-11-06', '2008-11-14'],level = '500 tx',
        wind_thres = 35,temp_thres = [40,20], rad_thres = [1100,350], sst_thres = [33,18], rh_thres=[120,0])
        data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=200)
plt_map_roll(hdtxx, data, txx, data_ttr,data_sst,dates = ['2008-10-27', '2008-11-01', '2008-11-05', '2008-11-09'],level = '200 tx',
        wind_thres = 70,temp_thres = [40,20], rad_thres = [1100,350], sst_thres = [32,18], rh_thres=[120,0])
        
        ####### tn ####
hdtnn = hwn_tn_roll.groupby('time.year').sum('time').max('year')
### Using 10 pctl analysis
l1=-22
l2=20
l3=30
l4=0
def plt_map_roll_tn(hd,data,temp,data_str,data_sst,dates,level,wind_thres,temp_thres, rad_thres, sst_thres,rh_thres):
    
    import matplotlib as mpl
    def show_axes():
        for i,j in enumerate(ax):
            ax[i].set_extent([-22,25,1,27])
            ax[i].add_feature(feature.COASTLINE)
            ax[i].add_feature(feature.BORDERS)
            ax[i].add_feature(feature.STATES, linewidth = 0.2)
            ax[i].set_xticks([-20,-10,0,10,20], crs=crs.PlateCarree())
            ax[i].set_yticks([10,20,27], crs=crs.PlateCarree())
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax[i].xaxis.set_major_formatter(lon_formatter)
            ax[i].yaxis.set_major_formatter(lat_formatter)
    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(15, 10),sharex=True, subplot_kw={'projection': crs.PlateCarree()})    
    
    ax = ax.flatten()
    show_axes()
    cmap = mpl.cm.rainbow
    cm = 'pink_r'
    # norm = mpl.colors.Normalize(vmax=wind_thres[0], vmin=wind_thres[1])
    # norm = mpl.colors.Normalize(vmin=0, vmax=int(wind_speed.max()))
    norm = mpl.colors.Normalize(vmin=0, vmax=wind_thres)
    # dates = ['1999-07-12', '1999-07-17', '1999-07-22', '1999-07-27']
    days = ['Prior 5 days', 'Day 1', 'Day 5', 'Post 5 days']

    for i in range(0,4):
        hwnp = hd.where(hd>0).plot(ax=ax[i],cbar_kwargs={'orientation': 'horizontal', 'label':'HWD'}, vmax = 11, cmap='bone_r')
        ax[i].set_title(days[i])

        
    for i, j in enumerate(dates):    
        rhp = data.r.sel(longitude=slice(l1,l2), latitude=slice(l3,l4)).sortby('latitude').sel(time=f'{dates[i]}T03', method='nearest').plot.contour(
            ax=ax[i+4],add_colorbar=False, cmap= 'rainbow', vmin=rh_thres[1], vmax=rh_thres[0])
        temp = tnn.sel(longitude=slice(l1,l2), latitude=slice(l4,l3)).sel(time=f'{dates[i]}', method = 'nearest').plot(
            ax=ax[i+4],cmap = cm,cbar_kwargs={'orientation': 'horizontal','label':'Temperature [$^o$C]'},vmax=temp_thres[0], vmin=temp_thres[1])
        radp = data_str.sel(longitude=slice(l1,l2), latitude=slice(l3,l4)).sortby('latitude').sel(
            time=f'{dates[i]}T03', method = 'nearest').plot(ax=ax[8+i],cbar_kwargs={
            'orientation': 'horizontal', 'label':'OLR [$Wm^2$C]'}, cmap= 'rainbow', vmax = rad_thres[0], vmin=rad_thres[1])
        sstp = data_sst.sel(longitude=slice(l1,l2), latitude=slice(l3,l4)).sortby('latitude').sel(time=f'{dates[i]}T03').plot(
            ax=ax[8+i],add_colorbar=False, cmap = 'jet', vmax=sst_thres[0], vmin=sst_thres[1])
        # ds = data.sel(longitude=slice(-15,10), latitude=slice(15,0)).sortby('latitude').drop(
        #     ['z','r','t','w','vo']).sel(time=f'{dates[i]}T03', method='nearest')
        # strm = ax[i].streamplot(x=ds["longitude"],y=ds["latitude"], u=ds["u"], v=ds["v"], color=ds.u.values, linewidth=1, cmap='rainbow', density=0.6, norm=norm)
        ds = data.sel(longitude=slice(l1,l2), latitude=slice(l3,l4)).sortby('latitude').drop(
            ['z','r','t','w','q']).sel(time=f'{dates[i]}T03', method='nearest')
        wind_speed = np.sqrt(ds.u[::3, ::3] ** 2 + ds.v[::3, ::3] ** 2)
        # norm = mpl.colors.Normalize(vmin=0, vmax=int(wind_speed.max()))
        quiver = ax[i].quiver(ds.u[::3,::3].longitude, ds.u[::3,::3].latitude, 
                               ds.u[::3,::3], ds.v[::3,::3],wind_speed, scale=1200-int(level[:-3]),
                              cmap='rainbow',linewidth=1.50, norm=norm, headwidth=5, headlength=5)

    for i in range(0,12):
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(None)

    cbmap = [rhp,sstp]#,strm]
    lb = ['Relative Humidity [%]','Sea Surface Temperature [$^o$C]']
    for i, j in enumerate([0.45, 0.13]):#, 0.18]):
        cax = fig.add_axes([1, j, 0.02, 0.15])
        fig.colorbar(cbmap[i], cax=cax, orientation='vertical', label = lb[i])

    cax = fig.add_axes([1, 0.77, 0.02, 0.15])
    # cbar = fig.colorbar(strm.lines, cax=cax, orientation='vertical', label = 'wind speed [ms$^-2$]')
    cbar = fig.colorbar(quiver, cax=cax, orientation='vertical', label='wind speed [ms$^-2$]')

    fig.suptitle(f'Pressure level = {level[:-3]} hpa', fontweight ='bold')
    plt.tight_layout()
    plt.savefig(f'/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/temp_wind_radiation_rh_hwn_sst_10day_pctl_tn_{level}{dates[0]}_1983-2016.jpeg', bbox_inches='tight')


data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=950)
plt_map_roll_tn(hdtnn, data, tnn, data_ttr,data_sst,dates = ['1988-10-04', '1988-10-09', '1988-10-14', '1988-10-24'],level = '950 tn',
        wind_thres = 20,temp_thres = [30,18], rad_thres = [1000,200], sst_thres = [30,23], rh_thres=[120,0])
data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=850)
plt_map_roll_tn(hdtnn, data, tnn, data_ttr,data_sst,dates = ['1988-10-04', '1988-10-09', '1988-10-14', '1988-10-24'],level = '850 tn',
        wind_thres = 20,temp_thres = [30,18], rad_thres = [1000,200], sst_thres = [30,23], rh_thres=[120,0])
        
data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=500)
plt_map_roll_tn(hdtnn, data, tnn, data_ttr,data_sst,dates = ['1988-10-04', '1988-10-09', '1988-10-14', '1988-10-24'],level = '500 tn',
        wind_thres = 26,temp_thres = [30,18], rad_thres = [1000,200], sst_thres = [30,23], rh_thres=[120,0])
        
data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').sel(level=200)
plt_map_roll_tn(hdtnn, data, tnn, data_ttr,data_sst,dates = ['1988-10-04', '1988-10-09', '1988-10-14', '1988-10-24'],level = '200 tn',
        wind_thres = 50,temp_thres = [30,18], rad_thres = [1000,200], sst_thres = [30,23], rh_thres=[120,0])
        
data = xr.open_dataset('/media/kenz/1B8D1A637BBA134B/Era_5/wind-tmp-rh.nc').chunk({'longitude':50,'latitude':50})

Cp = 1005
Lv = 2500840   #2.25*10**6
g = 9.8

MSE = data.q*Lv + data.t*Cp + data.z

MSE = MSE.compute()

MSE = MSE/1000

##### tn
plt.figure(figsize=(6,3))
tnn.sel(latitude=23, longitude=15, method='nearest').sel(time=slice('1988-10-04', '1988-10-24')).plot(alpha=0.3, label='Temperature')
region = tn.sel(latitude=23, longitude=15, method='nearest').sel(time=slice('1988-10-04', '1988-10-24'))
region.where(
    region.rolling(time=1).count()>=1
).plot(marker='*', alpha=0.3, label='Heat wave day')
region.where(
    region.rolling(time=11).count()>=11
).plot(marker='.')
plt.title('')
plt.grid()
plt.legend()
plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/hwd_tn_line.eps', bbox_inches='tight')

fig, ax = plt.subplots(ncols=4, nrows=2 , figsize=(10, 7), sharex=True, sharey=True)
ax = ax.flatten()
for i,j in enumerate(['2008-09-18-15', '2008-09-21', '2008-09-24', '2008-09-27']):
    MSE.sel(latitude=20, longitude=15, method='nearest').sel(time=f'{j}').plot(y='level', ax=ax[i],label='non heatwave day')
    
for i,j in enumerate(['2008-11-01-15', '2008-11-04', '2008-11-07', '2008-11-10']):
    plot2 = MSE.sel(latitude=20, longitude=15, method='nearest').sel(time=f'{j}').plot(y='level', ax=ax[i], label = 'heatwave day')
    # ax[i].set_title(j)

for i,j in enumerate(['1988-09-27-03', '1988-09-30', '1988-10-02', '1988-10-05']):
    MSE.sel(latitude=6.8, longitude=-5.2, method='nearest').sel(time=f'{j}').plot(y='level', ax=ax[i+4], label='non heatwave day')

grid = plt.GridSpec(2, 4)
create_subtitle(fig, grid[1, ::], 'Night Time \n')

for i,j in enumerate(['1988-10-09', '1988-10-12', '1988-10-15', '1988-10-18']):
    plot1 = MSE.sel(latitude=6.8, longitude=-5.2, method='nearest').sel(time=f'{j}').plot(y='level', ax=ax[i+4], label='heatwave day')
    # ax[i+4].set_title(j)

for i in range(0,8):
    ax[i].set_xlabel('MSE (kJ/kg)')
    ax[i].set_title('')
    
handles, labels = ax[i].get_legend_handles_labels()
plt.suptitle('Day Time', fontweight = 'bold')
plt.gca().invert_yaxis()
fig.legend(handles=handles,bbox_to_anchor=(1.175, 0.94))
plt.tight_layout()
plt.savefig('/media/kenz/1B8D1A637BBA134B/CHIRTS/final_figures/mse.eps', bbox_inches='tight')

