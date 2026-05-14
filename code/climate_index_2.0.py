# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:23:43 2026

@author: c337191
"""

from pathlib import Path
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import geopandas as gpd
import xarray as xr
import regionmask
import matplotlib.pyplot as plt
import warnings


# path
data_dir = Path(r'D:\Users\c337191\Documents\climate_migration\data\BR-DWGD')
path = r'D:\Users\c337191\Documents\climate_migration'

index_ds = xr.open_mfdataset(
    f"{path}/data/climate_indexes/climate_by_mesoregion.nc",
    combine="by_coords"
)
# Water balance, measures daily local aridity
index_ds["wb"] = index_ds["pr"] - index_ds["ETo"]

# %% functions for within indexes

# Within indexes
# For each realization of maximum and minimum temperature, aridity and flooding
# we compare the realization in a given day to the same day
# and a +-15 day window for the last 5 years, giving us 75 days as a base of comparison
# if the present day is a 5%pp occurence, it is labeled anomalous
# If weather phenomena are constant in time, we expect to label
# 5% of days (18) as anomalous every year
# Exposure is then the percentage of time a given region is under anomalous circumstances

def prepare_within_index_view(index_ds: xr.Dataset) -> xr.Dataset:
    """
    Return a lightweight view of index_ds with Feb 29 removed and with
    reusable calendar coordinates: year and doy365.

    This does not modify index_ds in place.
    """

    time = pd.DatetimeIndex(index_ds["time"].values)

    year = np.asarray(time.year, dtype=np.int16)
    dayofyear = np.asarray(time.dayofyear, dtype=np.int16)

    is_feb29 = (time.month == 2) & (time.day == 29)

    # Convert leap-year DOY into a 365-day climatological DOY.
    # In leap years, dates after Feb 29 have their day-of-year shifted down by 1.
    doy365 = dayofyear.copy()
    after_feb29_in_leap_year = np.asarray(time.is_leap_year) & (dayofyear > 59)
    doy365[after_feb29_in_leap_year] -= 1

    keep = ~np.asarray(is_feb29)

    return (
        index_ds
        .isel(time=keep)
        .assign_coords(
            year=("time", year[keep]),
            doy365=("time", doy365[keep]),
        )
    )

def _to_year_doy_region(da: xr.DataArray) -> xr.DataArray:
    """
    Reshape a daily DataArray from (time, region) into
    (year, doy365, region).
    Missing days, such as partial final years, become NaN.
    """
    years = np.arange(
        int(da["year"].min()),
        int(da["year"].max()) + 1,
        dtype=np.int16,
    )
    doys = np.arange(1, 366, dtype=np.int16)
    return (
        da.transpose("time", "region")
          .set_index(time=["year", "doy365"])
          .unstack("time")
          .reindex(year=years, doy365=doys)
          .transpose("year", "doy365", "region")
    )



def make_within_index(
    ds: xr.Dataset,
    var: str,
    *,
    tail: str = "upper",
    percentile: float = 0.95,
    baseline_years: int = 5,
    window: int = 15,
    min_ref_frac: float = 0.80,
    min_year_days: int | None = 300,
    rolling_days: int | None = None,
    rolling_stat: str = "sum",
    output_name: str | None = None,
    return_daily: bool = False,
) -> xr.Dataset:
    """
    Construct a within-region, seasonality-adjusted climate anomaly exposure index.

    For each region r and day t, the function compares x[r,t] to the empirical
    percentile of the same region's observations from:

        previous `baseline_years` years
        same climatological day ± `window` days

    Example with default settings:
        For 2000-01-15, compare to Jan 1--Jan 30 in 1995--1999.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dims ('time', 'region') and coords 'year', 'doy365'.
        Use prepare_within_index_view(index_ds) before calling this function.

    var : str
        Variable to process, e.g. 'Tmax', 'Tmin', 'pr', 'wb'.

    tail : {'upper', 'lower'}
        Which tail is bad.
        - 'upper': anomaly if x is unusually high.
        - 'lower': anomaly if x is unusually low.

    percentile : float
        Empirical percentile threshold.
        Use 0.95 for top 5 percent events.
        If tail='lower', 0.95 means bottom 5 percent of the original variable.

    baseline_years : int
        Number of past years used as comparison base.

    window : int
        Seasonal window around the same climatological day.
        window=15 means ±15 days, i.e. 31 days per baseline year.

    min_ref_frac : float
        Minimum fraction of the full reference window that must be non-missing.
        With 5 years and ±15 days, full reference size is 5 * 31 = 155.
        min_ref_frac=0.80 requires at least 124 valid reference observations.

    min_year_days : int or None
        Minimum number of classified days required to keep annual exposure.
        Use 300 to automatically drop partial years like 2024 if incomplete.
        Use None to keep partial-year exposures.

    rolling_days : int or None
        Optional pre-aggregation before anomaly calculation.
        Useful for:
        - precipitation: rolling_days=3 or 7
        - water balance/drought: rolling_days=30, 60, or 90

    rolling_stat : {'sum', 'mean'}
        How to aggregate when rolling_days is not None.

    output_name : str or None
        Prefix for output variable names.

    return_daily : bool
        If False, return only annual exposure and number of valid days.
        If True, also return the daily anomaly dummy.

    Returns
    -------
    xr.Dataset
        Compact dataset with variables:
        - {name}_exposure: annual share of anomalous days
        - {name}_n_days: number of classified days used in each annual exposure
        - optionally {name}_daily_anomaly
    """

    if var not in ds:
        raise KeyError(f"{var!r} not found in ds.")

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be either 'upper' or 'lower'.")

    if baseline_years < 1:
        raise ValueError("baseline_years must be >= 1.")

    if window < 0 or window > 182:
        raise ValueError("window must be between 0 and 182 for a 365-day calendar.")

    if rolling_days is not None and rolling_days < 1:
        raise ValueError("rolling_days must be None or >= 1.")

    q = percentile * 100 if 0 < percentile < 1 else percentile

    if not (0 < q < 100):
        raise ValueError("percentile must be in (0, 1) or (0, 100).")

    name = output_name or f"{var}_within"

    # Work with one variable only.
    # Casting to float32 reduces memory use and is more than enough here.
    x = ds[var].astype("float32")

    # Optional rolling aggregation.
    # For example:
    # - pr with rolling_days=7 gives 7-day accumulated precipitation.
    # - wb with rolling_days=90 gives 90-day accumulated water balance.
    if rolling_days is not None and rolling_days > 1:
        roller = x.rolling(time=rolling_days, min_periods=rolling_days)

        if rolling_stat == "sum":
            x = roller.sum()
        elif rolling_stat == "mean":
            x = roller.mean()
        else:
            raise ValueError("rolling_stat must be 'sum' or 'mean'.")

    # Convert lower-tail events into upper-tail events.
    # Example: drought is unusually low wb.
    # Using -wb means drought is unusually high -wb.
    if tail == "lower":
        x = -x

    # Reshape from time x region to year x doy365 x region.
    x_yday = _to_year_doy_region(x)

    # Intentional controlled compute:
    # This loads only one variable into memory, not the full dataset.
    # For your data, this is roughly 1961--2024 x 365 x 137,
    # especially manageable as float32.
    values = x_yday.compute().values.astype("float32", copy=False)

    years = x_yday["year"].values
    doys = x_yday["doy365"].values
    regions = x_yday["region"]

    n_years, n_doys, n_regions = values.shape

    if n_doys != 365:
        raise RuntimeError("Internal error: expected 365 day-of-year cells.")

    win_size = 2 * window + 1
    full_ref_days = baseline_years * win_size
    min_ref_days = int(np.ceil(min_ref_frac * full_ref_days))

    # Store daily anomaly as 1.0, 0.0, or NaN.
    # Float32 is used because boolean arrays cannot represent NaN.
    anomaly = np.full(values.shape, np.nan, dtype="float32")

    for iy in range(baseline_years, n_years):

        # Previous baseline_years only: no look-ahead.
        base = values[iy - baseline_years:iy, :, :]  # (baseline_years, 365, region)

        # Circular padding over day-of-year, so Jan 1 can compare to late December
        # from previous years, and Dec 31 can compare to early January.
        if window > 0:
            base_padded = np.concatenate(
                [base[:, -window:, :], base, base[:, :window, :]],
                axis=1,
            )

            # Shape:
            #   (baseline_years, 365, region, win_size)
            ref_windows = sliding_window_view(
                base_padded,
                window_shape=win_size,
                axis=1,
            )
        else:
            # Degenerate case: compare only exact same day-of-year.
            ref_windows = base[:, :, :, None]

        # Count valid reference observations for each target doy-region.
        valid_ref_count = np.isfinite(ref_windows).sum(axis=(0, 3))
        enough_ref = valid_ref_count >= min_ref_days

        # Empirical percentile over:
        #   baseline years x local seasonal window
        #
        # Result shape:
        #   (365, region)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            threshold = np.nanpercentile(
                ref_windows,
                q=q,
                axis=(0, 3),
            ).astype("float32")

        obs = values[iy, :, :]  # current year, all doy-region observations

        is_anom = (obs > threshold).astype("float32")

        # Mark as missing when:
        # - current observation missing
        # - insufficient reference data
        is_anom[(~np.isfinite(obs)) | (~enough_ref)] = np.nan

        anomaly[iy, :, :] = is_anom

    anom_yday = xr.DataArray(
        anomaly,
        coords={
            "year": years,
            "doy365": doys,
            "region": regions,
        },
        dims=("year", "doy365", "region"),
        name=f"{name}_daily_anomaly_yday",
    )

    exposure = anom_yday.mean("doy365", skipna=True).astype("float32")
    n_days = anom_yday.notnull().sum("doy365").astype("int16")

    # Drop partial or badly missing years from the exposure measure.
    # This is useful because your dataset appears to stop in March 2024.
    if min_year_days is not None:
        exposure = exposure.where(n_days >= min_year_days)

    exposure = exposure.rename(f"{name}_exposure")
    n_days = n_days.rename(f"{name}_n_days")

    out = xr.Dataset(
        {
            exposure.name: exposure,
            n_days.name: n_days,
        }
    )

    out[exposure.name].attrs.update(
        {
            "source_variable": var,
            "definition": (
                "Annual share of days whose value exceeds the empirical "
                "percentile of the same region's previous local seasonal windows."
            ),
            "tail": tail,
            "percentile": percentile,
            "baseline_years": baseline_years,
            "seasonal_window_plus_minus_days": window,
            "reference_days_if_complete": full_ref_days,
            "min_ref_days": min_ref_days,
            "min_year_days": min_year_days if min_year_days is not None else "None",
            "rolling_days": rolling_days if rolling_days is not None else "None",
            "rolling_stat": rolling_stat if rolling_days is not None else "None",
            "feb29": "dropped",
        }
    )

    if return_daily:
        # Map each original time observation in ds back into the year x doy365 array.
        year_to_pos = {int(y): i for i, y in enumerate(years)}

        year_pos = np.array(
            [year_to_pos[int(y)] for y in ds["year"].values],
            dtype=np.int16,
        )

        doy_pos = ds["doy365"].values.astype(np.int16) - 1

        daily = xr.DataArray(
            anomaly[year_pos, doy_pos, :],
            coords={
                "time": ds["time"],
                "region": regions,
            },
            dims=("time", "region"),
            name=f"{name}_daily_anomaly",
        )

        out[daily.name] = daily

    return out


def make_composite_exposure_within(
    daily_ds: xr.Dataset,
    components: list[str],
    *,
    name: str = "climate_within",
    require_all_components_valid: bool = True,
    min_year_days: int | None = 300,
) -> xr.Dataset:
    """
    Build a simple composite climate exposure index from daily anomaly dummies.

    The composite exposure is the annual share of days in which at least one
    climate anomaly is active.

    Parameters
    ----------
    daily_ds : xr.Dataset
        Dataset containing daily anomaly dummies with dims ('time', 'region').
        Each component should be 1 for anomalous, 0 for non-anomalous, and NaN
        when the day could not be classified.

    components : list[str]
        Names of daily anomaly variables inside daily_ds.

    name : str
        Prefix for output variable names.

    require_all_components_valid : bool
        If True, a day is valid only when all component dummies are non-missing.
        This gives a clean common denominator across dimensions.

        If False, a day is valid when at least one component is non-missing.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - {name}_exposure: share of valid days with at least one anomaly
        - {name}_n_days: number of valid days used in each region-year
    """

    if len(components) == 0:
        raise ValueError("components must contain at least one variable name.")

    for var in components:
        if var not in daily_ds:
            raise KeyError(f"{var!r} not found in daily_ds.")

    # Ensure a reusable year coordinate exists.
    if "year" not in daily_ds.coords or "time" not in daily_ds["year"].dims:
        years = np.asarray(pd.DatetimeIndex(daily_ds["time"].values).year, dtype=np.int16)
        daily_ds = daily_ds.assign_coords(year=("time", years))

    # Stack components into one array:
    # dims = ('component', 'time', 'region')
    arr = xr.concat(
        [daily_ds[var].astype("float32") for var in components],
        dim=xr.IndexVariable("component", components),
    ).transpose("component", "time", "region")

    # Define which day-region observations are usable.
    if require_all_components_valid:
        valid_day = arr.notnull().all("component")
    else:
        valid_day = arr.notnull().any("component")

    # A composite anomaly happens when at least one component is active.
    any_anomaly = (arr.fillna(0.0).sum("component") > 0)

    # Convert to 1/0/NaN, so invalid days are excluded from annual means.
    composite_daily = xr.where(
        valid_day,
        any_anomaly.astype("float32"),
        np.nan,
    )

    # Annual exposure: share of valid days with at least one anomaly.
    exposure = (
        composite_daily
        .groupby("year")
        .mean("time", skipna=True)
        .astype("float32")
        .rename(f"{name}_exposure")
    )

    # Number of classified days used in the denominator.
    n_days = (
        valid_day
        .groupby("year")
        .sum("time")
        .astype("int16")
        .rename(f"{name}_n_days")
    )
    if min_year_days is not None:
        exposure = exposure.where(n_days >= min_year_days)

    return xr.Dataset(
        {
            exposure.name: exposure,
            n_days.name: n_days,
        },
        attrs={
            "definition": (
                "Annual share of valid days with at least one active climate "
                "anomaly among the selected components."
            ),
            "components": ", ".join(components),
            "valid_day_rule": (
                "all components non-missing"
                if require_all_components_valid
                else "at least one component non-missing"
            ),
        },
    )

def drop_daily_vars(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars(
        [v for v in ds.data_vars if v.endswith("_daily_anomaly")]
    )

# %% Functions for between indexes

def make_between_index(
    ds: xr.Dataset,
    var: str,
    *,
    tail: str = "upper",
    percentile: float = 0.95,
    rolling_days: int | None = None,
    rolling_stat: str = "sum",
    min_regions: int = 100,
    min_year_days: int | None = 300,
    output_name: str | None = None,
    return_daily: bool = False,
) -> xr.Dataset:
    """
    Construct a between-region climate exposure index.

    For each day t, this compares each region's climate realization to the
    cross-sectional distribution of all regions on that same day.

    A region is exposed on day t if it lies in the bad national tail.

    Examples
    --------
    Tmax:
        exposed if Tmax_r,t is above the daily cross-region 95th percentile.

    Precipitation:
        exposed if rolling precipitation accumulation is above the daily
        cross-region 95th percentile.

    Water balance:
        exposed if water balance is below the daily cross-region 5th percentile.
        This is implemented by multiplying wb by -1 and then using the upper tail.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions ('time', 'region').

    var : str
        Variable to use, e.g. 'Tmax', 'Tmin', 'pr', 'wb'.

    tail : {'upper', 'lower'}
        Which tail is bad in the original variable.
        - 'upper': high values are bad.
        - 'lower': low values are bad.

    percentile : float
        Cross-sectional percentile threshold.
        Use 0.95 for the top 5 percent.

    rolling_days : int or None
        Optional rolling aggregation before ranking regions.
        Useful for:
        - pr: 3-day or 7-day rolling sum
        - wb: 60-day or 90-day rolling sum

    rolling_stat : {'sum', 'mean'}
        Rolling aggregation method.

    min_regions : int
        Minimum number of valid regions required on a day to compute the
        cross-sectional threshold.

    min_year_days : int or None
        Minimum number of valid daily classifications required to keep a
        region-year exposure value.

    output_name : str or None
        Prefix for output variables.

    return_daily : bool
        If True, also return the daily 0/1/NaN anomaly dummy.

    Returns
    -------
    xr.Dataset
        Compact annual dataset with:
        - {name}_exposure
        - {name}_n_days

        And, if return_daily=True:
        - {name}_daily_anomaly
    """

    if var not in ds:
        raise KeyError(f"{var!r} not found in ds.")

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be either 'upper' or 'lower'.")

    if rolling_days is not None and rolling_days < 1:
        raise ValueError("rolling_days must be None or >= 1.")

    q = percentile * 100 if 0 < percentile < 1 else percentile

    if not (0 < q < 100):
        raise ValueError("percentile must be in (0, 100) or (0, 1).")

    name = output_name or f"{var}_between"

    # Work with only one variable to avoid overloading memory.
    x = ds[var].astype("float32")

    # Optional rolling transformation.
    # This is strongly recommended for drought-like and flood-like objects.
    if rolling_days is not None and rolling_days > 1:
        roller = x.rolling(time=rolling_days, min_periods=rolling_days)

        if rolling_stat == "sum":
            x = roller.sum()
        elif rolling_stat == "mean":
            x = roller.mean()
        else:
            raise ValueError("rolling_stat must be either 'sum' or 'mean'.")

    # Convert lower-tail badness into upper-tail badness.
    # Example: low wb = dry, so -wb is high when conditions are dry.
    if tail == "lower":
        x = -x

    # Controlled compute:
    # this loads only one processed variable, not the whole dataset.
    x = x.transpose("time", "region")
    values = x.compute().values.astype("float32", copy=False)

    valid = np.isfinite(values)
    n_valid_regions = valid.sum(axis=1)

    # Compute one cross-sectional threshold per day.
    # threshold[t] = daily national q-th percentile.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        threshold = np.nanpercentile(values, q=q, axis=1).astype("float32")

    # Daily exposure dummy.
    anomaly = (values >= threshold[:, None]).astype("float32")

    # Missing if:
    # - region-day observation is missing
    # - too few regions are available to compute a credible daily percentile
    bad_day = n_valid_regions < min_regions
    anomaly[~valid] = np.nan
    anomaly[bad_day, :] = np.nan

    daily = xr.DataArray(
        anomaly,
        coords={
            "time": x["time"],
            "region": x["region"],
        },
        dims=("time", "region"),
        name=f"{name}_daily_anomaly",
    )

    # Add year coord for annual aggregation.
    if "year" in ds.coords and "time" in ds["year"].dims:
        daily = daily.assign_coords(year=("time", ds["year"].values))
    else:
        years = np.asarray(pd.DatetimeIndex(daily["time"].values).year, dtype=np.int16)
        daily = daily.assign_coords(year=("time", years))

    # Annual exposure: share of valid days in the national bad tail.
    exposure = (
        daily
        .groupby("year")
        .mean("time", skipna=True)
        .astype("float32")
        .rename(f"{name}_exposure")
    )

    # Number of classified days in each region-year.
    n_days = (
        daily
        .notnull()
        .groupby("year")
        .sum("time")
        .astype("int16")
        .rename(f"{name}_n_days")
    )

    if min_year_days is not None:
        exposure = exposure.where(n_days >= min_year_days)

    out = xr.Dataset(
        {
            exposure.name: exposure,
            n_days.name: n_days,
        }
    )

    out[exposure.name].attrs.update(
        {
            "source_variable": var,
            "definition": (
                "Annual share of days in which the region is in the national "
                "bad climate tail among all regions on the same date."
            ),
            "tail": tail,
            "percentile": percentile,
            "rolling_days": rolling_days if rolling_days is not None else "None",
            "rolling_stat": rolling_stat if rolling_days is not None else "None",
            "min_regions": min_regions,
            "min_year_days": min_year_days if min_year_days is not None else "None",
        }
    )

    if return_daily:
        out[daily.name] = daily

    return out


def make_composite_exposure_between(
    daily_ds: xr.Dataset,
    components: list[str],
    *,
    name: str = "climate_between_composite",
    min_year_days: int | None = 300,
    require_all_components_valid: bool = True,
) -> xr.Dataset:
    """
    Build a composite exposure index from daily anomaly dummies.

    The composite equals 1 on a region-day if at least one selected climate
    dimension is anomalous.

    Parameters
    ----------
    daily_ds : xr.Dataset
        Dataset containing daily 0/1/NaN anomaly dummies with dims
        ('time', 'region').

    components : list[str]
        Names of daily anomaly variables inside daily_ds.

    name : str
        Prefix for output variables.

    min_year_days : int or None
        Minimum valid days required to keep a region-year.

    require_all_components_valid : bool
        If True, a day is valid only if all component dummies are observed.
        This gives a common denominator across dimensions.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - {name}_exposure
        - {name}_n_days
    """

    if len(components) == 0:
        raise ValueError("components must contain at least one variable.")

    for var in components:
        if var not in daily_ds:
            raise KeyError(f"{var!r} not found in daily_ds.")

    # Ensure year coordinate exists.
    if "year" not in daily_ds.coords or "time" not in daily_ds["year"].dims:
        years = np.asarray(pd.DatetimeIndex(daily_ds["time"].values).year, dtype=np.int16)
        daily_ds = daily_ds.assign_coords(year=("time", years))

    # Stack components:
    # dims = component x time x region
    arr = xr.concat(
        [daily_ds[v].astype("float32") for v in components],
        dim=xr.IndexVariable("component", components),
    ).transpose("component", "time", "region")

    if require_all_components_valid:
        valid_day = arr.notnull().all("component")
    else:
        valid_day = arr.notnull().any("component")

    # Composite event: at least one climate dimension is active.
    any_anomaly = arr.fillna(0.0).sum("component") > 0

    composite_daily = xr.where(
        valid_day,
        any_anomaly.astype("float32"),
        np.nan,
    )

    exposure = (
        composite_daily
        .groupby("year")
        .mean("time", skipna=True)
        .astype("float32")
        .rename(f"{name}_exposure")
    )

    n_days = (
        valid_day
        .groupby("year")
        .sum("time")
        .astype("int16")
        .rename(f"{name}_n_days")
    )

    if min_year_days is not None:
        exposure = exposure.where(n_days >= min_year_days)

    return xr.Dataset(
        {
            exposure.name: exposure,
            n_days.name: n_days,
        },
        attrs={
            "definition": (
                "Annual share of valid days in which at least one selected "
                "climate dimension is in the national bad tail."
            ),
            "components": ", ".join(components),
            "min_year_days": min_year_days if min_year_days is not None else "None",
            "valid_day_rule": (
                "all components non-missing"
                if require_all_components_valid
                else "at least one component non-missing"
            ),
        },
    )

#%%
# Create this once and reuse it.
index_ds_365 = prepare_within_index_view(index_ds)
# Heat: unusually high daily maximum temperature
tmax_within = make_within_index(
    index_ds_365,
    "Tmax",
    tail="upper",
    min_year_days=300,
    output_name="Tmax_hot_within",
    return_daily= True
    )

# Cold: unusually low daily min temperature
tmin_within = make_within_index(
    index_ds_365,
    "Tmin",
    tail="lower",
    min_year_days=300,
    output_name="Tmin_cold_within",
    return_daily= True
    )

# Extreme precipitation.
# Raw daily pr is okay, but 3-day or 7-day accumulation is often cleaner.
pr_within = make_within_index(
    index_ds_365,
    "pr",
    tail="upper",
    min_year_days=300,
    output_name="pr_extreme_7d_within",
    rolling_days=7,
    rolling_stat="sum",
    return_daily= True
    )

# Drought/aridity: unusually low water balance.
# A rolling sum is strongly preferable for drought-like objects.
wb_within = make_within_index(
    index_ds_365,
    "wb",
    tail="lower",
    min_year_days=300,
    output_name="wb_dry_90d_within",
    rolling_days=90,
    rolling_stat="sum",
    return_daily= True
    )

# Computing composite within index
daily_anom_ds = xr.Dataset(
    {
        "Tmax_hot": tmax_within["Tmax_hot_daily_anomaly"],
        "Tmin_cold": tmin_within["Tmin_cold_daily_anomaly"],
        "pr_extreme_7d": pr_within["pr_extreme_7d_daily_anomaly"],
        "wb_dry_90d": wb_within["wb_dry_90d_daily_anomaly"],
    }
)

composite_ix_within = make_composite_exposure_within(
    daily_anom_ds,
    components=[
        "Tmax_hot",
        "Tmin_cold",
        "pr_extreme_7d",
        "wb_dry_90d",
    ],
    name="composite_within",
)
#%% Between indexes now
# Heat: regions in the hottest national tail on each day
tmax_between = make_between_index(
    index_ds_365,
    "Tmax",
    tail="upper",
    percentile=0.95,
    output_name="Tmax_hot_between",
    return_daily=True,
)

# Hot nights
tmin_between = make_between_index(
    index_ds_365,
    "Tmin",
    tail="lower",
    percentile=0.95,
    output_name="Tmin_cold_between",
    return_daily=True,
)

# Extreme precipitation: use accumulated rainfall
pr_between = make_between_index(
    index_ds_365,
    "pr",
    tail="upper",
    percentile=0.95,
    rolling_days=7,
    rolling_stat="sum",
    output_name="pr_extreme_7d_between",
    return_daily=True,
)

# Aridity/drought: low accumulated water balance
wb_between = make_between_index(
    index_ds_365,
    "wb",
    tail="lower",
    percentile=0.95,
    rolling_days=90,
    rolling_stat="sum",
    output_name="wb_dry_90d_between",
    return_daily=True,
)

daily_between_ds = xr.Dataset(
    {
        "Tmax_between_hot": tmax_between["Tmax_hot_between_daily_anomaly"],
        "Tmin_between_hot": tmin_between["Tmin_cold_between_daily_anomaly"],
        "pr_between_extreme_7d": pr_between["pr_extreme_7d_between_daily_anomaly"],
        "wb_between_dry_90d": wb_between["wb_dry_90d_between_daily_anomaly"],
    }
)

between_composite = make_composite_exposure_between(
    daily_between_ds,
    components=[
        "Tmax_between_hot",
        "Tmin_between_hot",
        "pr_between_extreme_7d",
        "wb_between_dry_90d",
    ],
    name="composite_between",
    min_year_days=300,
    require_all_components_valid=True,
)


# all index holder
relative_exposure_ds = xr.merge(
    [
        drop_daily_vars(tmax_within),
        drop_daily_vars(tmin_within),
        drop_daily_vars(pr_within),
        drop_daily_vars(wb_within),
        composite_ix_within,
        drop_daily_vars(tmax_between),
        drop_daily_vars(tmin_between),
        drop_daily_vars(pr_between),
        drop_daily_vars(wb_between),
        between_composite,
       ],
    compat="override",
)

n_day_vars = [v for v in relative_exposure_ds.data_vars if v.endswith("_n_days")]
relative_exposure_ds = relative_exposure_ds.drop_vars(n_day_vars)
relative_exposure_ds = relative_exposure_ds.drop_vars("time")
exposure_vars = list(relative_exposure_ds.data_vars)

relative_exposure_ds = relative_exposure_ds.dropna(
    dim="year",
    how="all",
    subset=exposure_vars
)

relative_exposure_ds.to_netcdf(f"{path}/data/climate_indexes/relative_exposure_index.nc")
df_rel = relative_exposure_ds.to_dataframe().reset_index().dropna()

df_rel.to_parquet(f"{path}/data/climate_indexes/relative_exposure_index.parquet", index=False)




# TODO ver a dispersão, estudar as correlações, e rodar umas regs aí







