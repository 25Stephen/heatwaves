import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cp
from cartopy import crs, feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import pymannkendall as mk
import warnings
warnings.filterwarnings('ignore')

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

fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (10,5), subplot_kw={'projection':crs.PlateCarree()})
ax = ax.flatten()
set_fig_params(ax)
# /media/kenz/1B8D1A637BBA134B/CHIRTS/Tmax/CHIRTS_Tmax_90.nc
tx90 = xr.open_dataset(path1+'/Tmax/chirts.Tmax90.1983.2016.WA.days_p25.nc').Tmax.sortby('time').drop_duplicates('time')
# /media/kenz/1B8D1A637BBA134B/CHIRTS/Tmean/chirts.Tmean.1983.2016.WA.days_p25.nc
tx = xr.open_dataset(path1+'/Tmax/chirts.Tmax.1983.2016.WA.days_p25.nc').Tmax.sortby('time').drop_duplicates('time')
mask = tx > tx90#.drop_duplicates('time')
ctx90 = tx.where(mask)
cc = 'gist_heat_r'
ctx90 = ctx90.groupby('time.year').count('time').mean('year').plot(ax = ax[0], vmax =2.5,
                                                                   cmap = cc, add_colorbar = False)
ax[0].set_title('CTx90 1983-2016')
tn90 = xr.open_dataset(path1+'/Tmin/chirts.Tmin90.1983.2016.WA.days_p25.nc').Tmin.sortby('time').drop_duplicates('time')
tn = xr.open_dataset(path1+'/Tmin/chirts.Tmin.1983.2016.WA.days_p25.nc').Tmin.sortby('time').drop_duplicates('time')
mask = tn > tn90
ctn90 = tn.where(mask)
ctn90 = ctn90.groupby('time.year').count('time').mean('year').plot(ax = ax[1],vmax=2.5, 
                                                                   cmap = cc, add_colorbar = False)

ax[1].set_title('CTn90 1983-2016')

EHF = xr.open_dataset('scripts/EHF.nc')

cb1 = abs(EHF.Tmax.mean('time')).plot(ax = ax[2], cmap = cc,vmax = 2.5, add_colorbar=False)#, extend = 'both', vmax = 2, shrink = 0.8)
ax[2].set_title('EHF 1983-2016')
cm = 'RdBu_r'
tx90_s = sens_slope(tx90)
tx90_s.plot(ax = ax[3], cmap = cm, add_colorbar=False)
ax[3].set_title('tx90 sens slope')

tn90_s = sens_slope(tn90)
cb2 = tn90_s.plot(ax = ax[4], cmap = cm, add_colorbar=False)
ax[4].set_title('tn90 sens slope')


ehf_s = sens_slope(EHF.Tmax)
ehf_s.plot(ax = ax[5], cmap = cm, add_colorbar=False)
ax[5].set_title('ehf sens slope')
cb = [cb1,cb2]
for i,j in enumerate([0.62,0.13]):
    cax = fig.add_axes([1,j,0.02,0.3])
    fig.colorbar(cb[i], cax=cax, orientation='vertical', extend = 'both')
# fig.colorbar(shrink=0.8)
### plt.suptitle('Climatologies and trends for HWN')
fig.tight_layout()
plt.savefig(path1+'/figures/graph1.jpeg', bbox_inches='tight')

# tx90 = xr.open_dataset(path1+'/Tmax/chirts.Tmax90.1983.2016.WA.days_p25.nc').Tmax.sortby('time').drop_duplicates('time')
tx = xr.open_dataset(path1+'/Tmax/chirts.Tmax.1983.2016.WA.days_p25.nc').Tmax.sortby('time').drop_duplicates('time').chunk({'time':20})
tn = xr.open_dataset(path1+'/Tmin/chirts.Tmin.1983.2016.WA.days_p25.nc').Tmin.sortby('time').drop_duplicates('time').chunk({'time':20})

fig, ax = plt.subplots(ncols = 3, nrows=1, figsize = (10,5), subplot_kw={'projection':crs.PlateCarree()})
ax = ax.flatten()
set_fig_params(ax)
vmax = 8
cm = 'YlOrRd'
mask = xr.open_dataset('Tmax/tx-tx90.nc').chunk({'time':20}).Tmax
mask = HWD(mask)
cb = (mask.sel(time=slice('1983','2016')).sum('time')/30).plot(ax=ax[0],cmap = cm, add_colorbar=False, vmax = vmax) 
ax[1].set_title('Tmax 1983 2016')

mask = xr.open_dataset('Tmin/tn-tn90.nc').chunk({'time':20}).Tmin
mask = HWD(mask)
(mask.sel(time=slice('1983','2016')).sum('time')/30).plot(ax=ax[1],cmap = cm, add_colorbar=False, vmax=vmax) 
ax[1].set_title('Tmin 1983 2016')

mask = xr.open_dataset('Tmean/tm-tm90.nc').chunk({'time':20}).Tmax
mask = HWD(mask)
(mask.sel(time=slice('1983','2016')).sum('time')/30).plot(ax=ax[2],cmap = cm, add_colorbar=False, vmax=vmax) 
ax[2].set_title('Tmean 1983 2016')

cax = fig.add_axes([1,0.35,0.02,0.3])
fig.colorbar(cb, cax=cax, orientation='vertical', extend = 'both')
# # fig.colorbar(shrink=0.8)
# ### plt.suptitle('Climatologies and trends for HWN')
fig.tight_layout()
plt.savefig(path1+'/figures/graph2.jpeg', bbox_inches='tight')
