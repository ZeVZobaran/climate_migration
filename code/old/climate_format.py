# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:38:52 2026

@author: c337191
"""
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import regionmask
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# path
data_dir = Path(r'D:\Users\c337191\Documents\EnvEcon\data\BR-DWGD')
path = r'D:\Users\c337191\Documents\EnvEcon'
#%% opening files

regions = gpd.read_file(f'{path}/data/br_mesorregioes/BRMEE250GC_SIR.shp')
# Climate grids are in lon/lat, so polygons should also be in EPSG:4326
regions = regions.to_crs("EPSG:4326").copy()
regions['CD_GEOCME'] = pd.to_numeric(regions['CD_GEOCME'])

all_nc_files = sorted(data_dir.glob("*.nc"))

# Group files by variable name based on filename prefix
file_groups = {
    "pr":   sorted([f for f in all_nc_files if f.name.startswith("pr_")]),
    "Tmax": sorted([f for f in all_nc_files if f.name.startswith("Tmax_")]),
    "Tmin": sorted([f for f in all_nc_files if f.name.startswith("Tmin_")]),
    "ETo":  sorted([f for f in all_nc_files if f.name.startswith("ETo_")])
}


def open_variable_dataset(file_list, var_name):
    """
    Opens several NetCDF files belonging to the same variable and concatenates
    them along time into one xarray Dataset.

    Parameters
    ----------
    file_list : list of pathlib.Path
        The .nc files for one climate variable.
    var_name : str
        Variable name expected inside the NetCDF files, e.g. 'pr', 'Tmax', 'Tmin'.

    Returns
    -------
    ds : xarray.Dataset
        Combined dataset sorted by time, with duplicate dates removed if needed.
    """
    if len(file_list) == 0:
        raise ValueError(f"No files found for variable {var_name}")

    ds = xr.open_mfdataset(
        [str(f) for f in file_list],
        combine="by_coords"
    )

    # Ensure time is sorted
    ds = ds.sortby("time")

    # Keep only unique time stamps, in case files overlap slightly
    _, unique_idx = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=np.sort(unique_idx))

    # Keep only the target variable if extra vars exist
    ds = ds[[var_name]]

    return ds

climate_ds = {}

for var_name, files in file_groups.items():
    climate_ds[var_name] = open_variable_dataset(files, var_name)

# %% setting up masks and weights

meso_mask = regionmask.from_geopandas(
    regions,
    numbers="CD_GEOCME",   # unique code for each mesoregion
    names="NM_MESO"        # human-readable region name
    )

# using the precipitation dataset for masks
reference_var = "pr"
ref_ds = climate_ds[reference_var]

mask_3d = meso_mask.mask_3D(
    ref_ds.longitude, ref_ds.latitude) # mask_3d dims -> ('region', 'latitude', 'longitude')

# data on a lon-lat grid --> cells not equally sized:
# cells closer to the poles are physically smaller.
# A common approximation is to weight by cos(latitude).
lat_weights = np.cos(np.deg2rad(ref_ds["latitude"]))
weights_2d = lat_weights.broadcast_like(ref_ds["pr"].isel(time=0))

# %% aggregating variables to mesorregion

def aggregate_to_regions(ds, var_name, mask_3d, weights_2d, regions_gdf):
    """
    Computes area-weighted daily averages of one climate variable for each mesoregion.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dimensions (time, latitude, longitude)
    var_name : str
        Name of the climate variable in ds
    mask_3d : xarray.DataArray
        Boolean mask with dims (region, latitude, longitude)
    weights_2d : xarray.DataArray
        Area weights with dims (latitude, longitude)
    regions_gdf : GeoDataFrame
        Mesoregions with CD_GEOCME and NM_MESO

    Returns
    -------
    da_region : xarray.DataArray
        Aggregated variable with dims (time, region)
    """
    da = ds[var_name]

    # Weighted sum over all cells inside each region
    weighted_sum = (da.where(mask_3d) * weights_2d.where(mask_3d)).sum(
        dim=("latitude", "longitude")
    )

    # Sum of weights inside each region
    sum_of_weights = weights_2d.where(mask_3d).sum(
        dim=("latitude", "longitude")
    )

    # Area-weighted regional mean
    da_region = weighted_sum / sum_of_weights

    # Attach readable region names
    region_lookup = (
        regions_gdf[["CD_GEOCME", "NM_MESO"]]
        .drop_duplicates()
        .set_index("CD_GEOCME")
    )

    da_region = da_region.assign_coords(
        NM_MESO=("region", region_lookup.loc[da_region["region"].values, "NM_MESO"].values)
    )

    da_region.name = var_name
    return da_region

regional_dataarrays = []

for var_name, ds in climate_ds.items():
    da_region = aggregate_to_regions(
        ds=ds,
        var_name=var_name,
        mask_3d=mask_3d,
        weights_2d=weights_2d,
        regions_gdf=regions
    )
    regional_dataarrays.append(da_region)

# %% output objs

climate_region_ds = xr.merge([da.to_dataset(name=da.name) for da in regional_dataarrays])
climate_region_ds = climate_region_ds.chunk({"time": 3650})

# This will have roughly:
# dimensions: (time, region)
# coordinates: time, region, NM_MESO
# variables: pr, Tmax, Tmin, ETo

#climate_region_df = climate_region_ds.to_dataframe().reset_index()

# Optional: reorder columns nicely
#first_cols = ["time", "region", "NM_MESO"]
#other_cols = [c for c in climate_region_df.columns if c not in first_cols]
#climate_region_df = climate_region_df[first_cols + other_cols]

#climate_region_df.to_parquet(data_dir / "climate_by_mesoregion.parquet", index=False)

# %% setting up indexes

# reinstanciate the ds for performance (one hopes)
index_ds = xr.open_mfdataset(
    f"{path}/data/climate_indexes/climate_by_mesoregion.nc",
    combine="by_coords"
)
# temperature mean

index_ds ["Tmean"] = (index_ds ["Tmax"] + index_ds ["Tmin"]) / 2
# we define the "good" temperature as 23C
# the farther away from this you are, the worse
t_opt = 23
index_ds["comfort_dev"] = (index_ds ["Tmean"] - t_opt) ** 2

# extreme event index
# heatwave: days above summer average + 4.5 (Hernandez-Cortez and mathes, 2024)
# summer-years (as summer spans two different years)
month = index_ds["time"].dt.month
year = index_ds["time"].dt.year
summer_year = xr.where(month == 12, year + 1, year)
index_ds  = index_ds.assign_coords(summer_year=("time", summer_year.data))

summer_mask = month.isin([12, 1, 2])
ds_summer = index_ds.sel(time=summer_mask)
summer_avg = ds_summer["Tmean"].groupby("summer_year").mean("time")
# Expanding baseline: we define heatwaves against past and current summer only
summer_baseline = summer_avg.where(summer_avg["summer_year"] < 2000, drop=True).mean("summer_year")
baseline_year = summer_year
index_ds = index_ds.assign_coords(baseline_year=("time", baseline_year.data))

# Map baseline to each day
heat_threshold = summer_baseline + 4.5
# Heat event on any day of the year
heat_event = index_ds["Tmean"] > heat_threshold

# Annual exposure: share of days classified as heat events
heat_exposure = heat_event.groupby("time.year").mean("time")
heat_exposure.name = 'heat_exposure'

# floods: days with precipitation > 55mm (hernandez-Cortez and Mathes, 2024)
flood_event = index_ds["pr"] > 55
flood_exposure = flood_event.groupby("time.year").mean("time")
flood_exposure.name = 'flood_exposure'

# drought anomalies: SPEI index
# 1. Daily climatic water balance
index_ds["wb"] = index_ds["pr"] - index_ds["ETo"]
# 2. Rolling accumulated balance
k = 90
wb_roll = index_ds["wb"].rolling(time=k, min_periods=k).sum()
# 3. Day of year
wb_roll = wb_roll.assign_coords(doy=("time", wb_roll["time"].dt.dayofyear.data))
# 4. Climatology by day-of-year and region
wb_mu = wb_roll.groupby("doy").mean("time")
wb_sigma = wb_roll.groupby("doy").std("time")
# 5. Standardized water-balance drought index
sepi_90 = (wb_roll.groupby("doy") - wb_mu) / wb_sigma
# drop the DOY dimension as it is no longer useful
sepi_90 = sepi_90.sel(doy=sepi_90["time"].dt.dayofyear).drop_vars('doy')
del wb_roll
del wb_mu
del wb_sigma
# 6. Drought event definition
drought_event = sepi_90 < -1.5  # worse than 1.5 sigmas --> drought
del sepi_90
#drought_event.sum().compute().item()
# 7. Annual drought exposure
drought_exposure = drought_event.groupby("time.year").mean("time")
drought_exposure.name = "drought_exposure"
del drought_event

# high aridity / seasonal droughts:
# I compare regional water balance to mean national water balance for this
# Use the same k as for the SPEI index for coherency
wb_roll = index_ds["wb"].rolling(time=k, min_periods=k).sum()
wb_space_z = (
    wb_roll - wb_roll.mean("region")
) / wb_roll.std("region")
wb_space_z = wb_space_z.drop_vars(['summer_year', 'baseline_year'])
drought_abs_event = wb_space_z < -1.5  # worse than 1.5 sigmas --> locallly drought
drought_abs_exposure = drought_abs_event.groupby("time.year").mean("time")
drought_abs_exposure.name = "drought_abs_exposure"

# final event index
exposure_ds = xr.Dataset({
    "heat": heat_exposure,
    "flood": flood_exposure,
    "drought_anomaly": drought_exposure,  # drought relative to region historic
    'drought_absolute': drought_abs_exposure # drought relative to current country status
})


# temperature stability
stability_ds = -index_ds["Tmean"].groupby("time.year").var("time")
# temperature mean
livability_ds = -index_ds["comfort_dev"].groupby("time.year").mean("time")

del index_ds
del ds_summer
del ds
del ref_ds
del baseline_year
del da_region
del drought_exposure
del flood_event
del flood_exposure
del heat_event
del heat_exposure
del heat_threshold
del year
del summer_mask
del month
del mask_3d
del summer_year
del weights_2d
del summer_avg
del summer_baseline

# %% saving and gen DFs

climate_region_ds.to_netcdf(f"{path}/data/climate_indexes/climate_by_mesoregion.nc")
climate_region_df = climate_region_ds.to_dataframe().reset_index()
del climate_region_ds
stability_ds.to_netcdf(f"{path}/data/climate_indexes/stability.nc")
stability_df = stability_ds.to_dataframe().reset_index()
del stability_ds
livability_ds.to_netcdf(f"{path}/data/climate_indexes/livability.nc")
livability_df = livability_ds.to_dataframe().reset_index()
del livability_ds

# %% final snippet to generate z scores

ds_z = xr.Dataset()
# Yearly z score of how much heat / drought / flood each place got
# as compared to the rest of the country on that same year
for v in ["heat","drought_anomaly","flood", 'drought_absolute']:
    mu = exposure_ds[v].mean("region")
    sigma = exposure_ds[v].std("region")
    ds_z[v] = (exposure_ds[v] - mu) / sigma

# final index
ds_z["exposure_index"] = (
    ds_z["heat"] +
    ds_z["drought_anomaly"] +
    ds_z["flood"] + 
    ds_z["drought_absolute"]
) / 4

ds_z.to_netcdf(f"{path}/data/climate_indexes/exposure_zscore.nc")
df_z = ds_z.to_dataframe().reset_index()
df_z.to_parquet(f"{path}/data/climate_indexes/exposure_z_scores.parquet", index=False)
exposure_df = exposure_ds.to_dataframe().reset_index()
exposure_df.to_parquet(f"{path}/data/climate_indexes/exposure.parquet", index=False)
exposure_ds.to_netcdf(f"{path}/data/climate_indexes/exposure.nc")


