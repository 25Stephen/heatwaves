%%time
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cp
from cartopy import crs, feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import pymannkendall as mk
# from cdo import *
import warnings
warnings.filterwarnings('ignore')
# cdo = Cdo()

path1='/media/kenz/1B8D1A637BBA134B/CHIRTS'

def set_fig_params(axes):
    for i,j in enumerate(axes):
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

        
def sens_slope(data):
    data = data.groupby('time.year').mean('time')
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

def rising_filter(array, axis):

    # Make sure there are enough points
    assert(array.shape[axis] == 5)
    # Make sure we're working on the last axis
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
    s = data>0
    s = s.drop_duplicates('time')
    candidates = tx.where(s)
    windows = candidates.chunk({'time':20}).rolling(time=5, center=True, min_periods=1).construct('rolling_dim')
    heatwave_starts = rising_filter_dask(windows, dim='rolling_dim')
    return heatwave_starts

%%time

def cal_ct90(path1, temperature_type='Tmax', return_var=None):
    if temperature_type not in ['Tmax', 'Tmin']:
        raise ValueError("Invalid temperature_type. Use 'Tmax' or 'Tmin'.")
    # if return_var not in [

    # Load temperature data based on the temperature_type
    t90 = xr.open_dataset(f'{path1}/{temperature_type}/chirts.{temperature_type}90.1983.2016.WA.days_p25.nc').sortby('time').drop_duplicates('time')[f'{temperature_type}']
    t = xr.open_dataset(f'{path1}/{temperature_type}/chirts.{temperature_type}.1983.2016.WA.days_p25.nc').sortby('time').drop_duplicates('time')[f'{temperature_type}']

    if return_var == 'T':
        return t
    
    elif return_var == 'T90':
        return t90
    
    else:
        # Calculate mask for ct90
        mask = t > t90
        ct90 = t.where(mask)
        return ct90
# def df_read(path,file):
#     ds = xr.open_dataset(f'{path}/{file}').Tmax
#     return ds

# def HWF(path,file, temperature_type='Tmax'):
#     if temperature_type not in ['Tmax', 'Tmin']:
#         raise ValueError("Invalid temperature_type. Use 'Tmax' or 'Tmin'.")
#     path,file, temperature_type = path,file, temperature_type
#     ds = df_read(path,file)
#     ds = HWD(ds).sel(time=slice('1983', '2016')).sum('time') / 30
#     ds = cal_ct90(path, temperature_type=temperature_type).groupby('time.year').count('time').mean('year')*ds
#     return ds

# def HWF_ehf(path, file):
#     ds = df_read(path, file).sortby('time').chunk({'time':5})
#     mask = ds.where(ds>0, drop = True)
#     ehf_hwd = mask.rolling(time=3).sum().sel(time=slice('1983', '2016')).mean('time')
#     ehf = abs(ds.mean('time'))
#     return ehf*ehf_hwd

def df_read(path,file,var):
    ds = xr.open_dataset(f'{path}/{file}')[f'{var}']
    # [f'{temperature_type}']
    return ds

def HWF(path,file, temperature_type='Tmax',var='Tmax'):
    if temperature_type not in ['Tmax', 'Tmin']:
        raise ValueError("Invalid temperature_type. Use 'Tmax' or 'Tmin'.")
    path,file, temperature_type = path,file, temperature_type
    ds = df_read(path,file,var)
    ds = HWD(ds).sel(time=slice('1983', '2016')).sum('time') / 30
    ds = cal_ct90(path, temperature_type=temperature_type).groupby('time.year').count('time').mean('year')*ds
    return ds

def HWF_ehf(path, file, var):
    ds = df_read(path, file,var).sortby('time').chunk({'time':5})
    mask = ds.where(ds>0, drop = True)
    ehf_hwd = mask.rolling(time=3).sum().sel(time=slice('1983', '2016')).mean('time')
    ehf = abs(ds.mean('time'))
    return ehf*ehf_hwd

%%time

# Create subplots
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})
ax = ax.flatten()
set_fig_params(ax)


# Plot data
cc = 'gist_heat_r'
cb1 = cal_ct90(path1, temperature_type='Tmax').groupby('time.year').count('time').mean('year').plot(ax=ax[0], vmax=2.5, cmap=cc, add_colorbar=False)
ax[0].set_title('CTx90 1983-2016')

# # cc = 'gist_heat_r'
cal_ct90(path1, temperature_type='Tmin').groupby('time.year').count('time').mean('year').plot(ax=ax[1], vmax=2.5, cmap=cc, add_colorbar=False)
ax[1].set_title('CTn90 1983-2016')

# EHF = xr.open_dataset(f'{path1}/scripts/EHF.nc')
abs(df_read(path1, 'scripts/EHF.nc', var = 'Tmax')).mean('time').plot(ax=ax[2], cmap=cc, vmax=2.5, add_colorbar=False)

# cb1 = abs(EHF.Tmax.mean('time')).plot(ax=ax[2], cmap=cc, vmax=2.5, add_colorbar=False)
ax[2].set_title('EHF 1983-2016')

cm = 'RdBu_r'
# tx90_s = sens_slope(tx90)
cb2 = sens_slope(cal_ct90(path1, temperature_type='Tmax', return_var='T90')).plot(ax=ax[3], cmap=cm, add_colorbar=False)
ax[3].set_title('tx90 sens slope')

# tn90_s = sens_slope(tn90)
sens_slope(cal_ct90(path1, temperature_type='Tmin', return_var='T90')).plot(ax=ax[4], cmap=cm, add_colorbar=False)
# cb2 = tn90_s.plot(ax=ax[4], cmap=cm, add_colorbar=False)
ax[4].set_title('tn90 sens slope')

# ehf_s = sens_slope(EHF.Tmax)
sens_slope(abs(df_read(path1, 'scripts/EHF.nc', var = 'Tmax'))).plot(ax=ax[5], cmap=cm, add_colorbar=False)
# ehf_s.plot(ax=ax[5], cmap=cm, add_colorbar=False)
ax[5].set_title('ehf sens slope')

cb = [cb1, cb2]
label = ['HWN','Trends']

# Add colorbars
for i, j in enumerate([0.62, 0.13]):
    cax = fig.add_axes([1, j, 0.02, 0.3])
    fig.colorbar(cb[i], cax=cax, orientation='vertical', extend='both', label = label[i])

fig.tight_layout()
# # plt.savefig(path1+'/figures/graph1.jpeg', bbox_inches='tight')

%%time

# Create subplots
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})
ax = ax.flatten()
set_fig_params(ax)


# Plot data
cc = 'gist_heat_r'
cb1 = cal_ct90(path1, temperature_type='Tmax').groupby('time.year').count('time').mean('year').plot(ax=ax[0], vmax=2.5, cmap=cc, add_colorbar=False)
ax[0].set_title('CTx90 1983-2016')

# # cc = 'gist_heat_r'
cal_ct90(path1, temperature_type='Tmin').groupby('time.year').count('time').mean('year').plot(ax=ax[1], vmax=2.5, cmap=cc, add_colorbar=False)
ax[1].set_title('CTn90 1983-2016')

# EHF = xr.open_dataset(f'{path1}/scripts/EHF.nc')
abs(df_read(path1, 'scripts/EHF.nc', var = 'Tmax')).mean('time').plot(ax=ax[2], cmap=cc, vmax=2.5, add_colorbar=False)

# cb1 = abs(EHF.Tmax.mean('time')).plot(ax=ax[2], cmap=cc, vmax=2.5, add_colorbar=False)
ax[2].set_title('EHF 1983-2016')

cm = 'RdBu_r'
# tx90_s = sens_slope(tx90)
cb2 = sens_slope(cal_ct90(path1, temperature_type='Tmax', return_var='T90')).plot(ax=ax[3], cmap=cm, add_colorbar=False)
ax[3].set_title('tx90 sens slope')

# tn90_s = sens_slope(tn90)
sens_slope(cal_ct90(path1, temperature_type='Tmin', return_var='T90')).plot(ax=ax[4], cmap=cm, add_colorbar=False)
# cb2 = tn90_s.plot(ax=ax[4], cmap=cm, add_colorbar=False)
ax[4].set_title('tn90 sens slope')

# ehf_s = sens_slope(EHF.Tmax)
sens_slope(abs(df_read(path1, 'scripts/EHF.nc', var = 'Tmax'))).plot(ax=ax[5], cmap=cm, add_colorbar=False)
# ehf_s.plot(ax=ax[5], cmap=cm, add_colorbar=False)
ax[5].set_title('ehf sens slope')

cb = [cb1, cb2]
label = ['HWN','Trends']

# Add colorbars
for i, j in enumerate([0.62, 0.13]):
    cax = fig.add_axes([1, j, 0.02, 0.3])
    fig.colorbar(cb[i], cax=cax, orientation='vertical', extend='both', label = label[i])

fig.tight_layout()
# # plt.savefig(path1+'/figures/graph1.jpeg', bbox_inches='tight')

%%time

# Load and process your data
tx = xr.open_dataset(f'{path1}/Tmax/chirts.Tmax.1983.2016.WA.days_p25.nc').Tmax.sortby('time').drop_duplicates('time').chunk({'time': 20})
tn = xr.open_dataset(f'{path1}/Tmin/chirts.Tmin.1983.2016.WA.days_p25.nc').Tmin.sortby('time').drop_duplicates('time').chunk({'time': 20})

# Create subplots
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})
ax = ax.flatten()
set_fig_params(ax)

vmax = 8
vmax1 = 0.01
vmin1 = -0.01

cm = 'YlOrRd'
cc = 'rainbow'
# Calculate and plot Tmax
mask = xr.open_dataset(f'{path1}/Tmax/tx-tx90.nc').Tmax
sens_slope(mask).plot(ax=ax[3], cmap=cc, add_colorbar=False, vmax = vmax1, vmin=vmin1)
mask = mask.chunk({'time': 20})
tx_tx90 = HWD(mask)
cb1 = (tx_tx90.sel(time=slice('1983', '2016')).sum('time') / 30).plot(ax=ax[0], cmap=cm, add_colorbar=False, vmax=vmax)
ax[0].set_title('CTx90 \n 1983-2016')

# Calculate and plot Tmin
mask = xr.open_dataset(f'{path1}/Tmin/tn-tn90.nc').Tmin
sens_slope(mask).plot(ax=ax[4], cmap=cc, add_colorbar=False, vmax = vmax1, vmin=vmin1)
mask = mask.chunk({'time': 20})
tn_tn90 = HWD(mask)
(tn_tn90.sel(time=slice('1983', '2016')).sum('time') / 30).plot(ax=ax[1], cmap=cm, add_colorbar=False, vmax=vmax)
ax[1].set_title('CTn90 \n 1983-2016')

# Calculate and plot EHF
mask = xr.open_dataset(f'{path1}/scripts/EHF.nc').sortby('time').drop('time_bnds').Tmax
cb2 = sens_slope(mask).plot(ax=ax[5], cmap=cc, add_colorbar=False, vmax = vmax1, vmin = vmin1)

mask = mask.where(mask > 0, drop=True)
ehf_hwd = mask.rolling(time=3).sum()
(ehf_hwd.sel(time=slice('1983', '2016')).mean('time')).plot(ax=ax[2], cmap=cm, add_colorbar=False, vmax=vmax)
ax[2].set_title('EHF \n 1983-2016')

# ax[3].set_title('EHF \n 1983-2016')
# ax[4].set_title('EHF \n 1983-2016')
# ax[5].set_title('EHF \n 1983-2016')

# Add colorbar
cb = [cb1, cb2]
labels = ['HWD','Trends']
# Add colorbars
for i, j in enumerate([0.585, 0.13]):
    cax = fig.add_axes([1, j, 0.02, 0.3])
    fig.colorbar(cb[i], cax=cax, orientation='vertical', extend='both', label=labels[i])
    
# cax = fig.add_axes([1, 0.35, 0.02, 0.3])
# fig.colorbar(cb, cax=cax, orientation='vertical', extend='both', label='HWD')

fig.tight_layout()
# plt.savefig(path1+'/figures/graph2.jpeg', bbox_inches='tight')



tx = xr.open_dataset(f'{path1}/Tmax/chirts.Tmax.1983.2016.WA.days_p25.nc').Tmax.sortby('time').drop_duplicates('time').chunk({'time': 20})
tn = xr.open_dataset(f'{path1}/Tmin/chirts.Tmin.1983.2016.WA.days_p25.nc').Tmin.sortby('time').drop_duplicates('time').chunk({'time': 20})

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})
ax = ax.flatten()
set_fig_params(ax)

vmax = 8

cm = 'viridis'
cc = 'spring'

HWF(path1, 'Tmax/tx-tx90.nc', temperature_type='Tmax', var = 'Tmax').plot(ax=ax[0], cmap=cc, vmax=vmax)


HWF(path1, 'Tmin/tn-tn90.nc', temperature_type='Tmin', var = 'Tmin').plot(ax=ax[1], cmap=cc, vmax=vmax)


HWF_ehf(path1, 'scripts/EHF.nc', var = 'Tmax').plot(ax=ax[2], cmap=cc)

ax[2].set_title('EHF \n 1983-2016')



# sens_slope(HWF(path1, 'Tmax/tx-tx90.nc', temperature_type='Tmax', var = 'Tmax')).plot(ax=ax[3], cmap=cc)
# sens_slope(HWF(path1, 'Tmin/tn-tn90.nc', temperature_type='Tmin', var = 'Tmin')).plot(ax=ax[4], cmap=cc)
# sens_slope(HWF_ehf(path1, 'scripts/EHF.nc', var = 'Tmax')).plot(ax=ax[5], cmap=cc)

# ax[3].set_title('EHF \n 1983-2016')
# ax[4].set_title('EHF \n 1983-2016')
# ax[5].set_title('EHF \n 1983-2016')

# cb = [cb1, cb2]
# labels = ['HWF','Trends']
# # Add colorbars
# for i, j in enumerate([0.59, 0.13]):
#     cax = fig.add_axes([1, j, 0.02, 0.3])
#     fig.colorbar(cb[i], cax=cax, orientation='vertical', extend='both', label=labels[i])

fig.tight_layout()
plt.savefig(path1+'/figures/graph3.jpeg', bbox_inches='tight')
