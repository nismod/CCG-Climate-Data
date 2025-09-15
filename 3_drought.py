import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spei as si
import xarray as xr
from scipy.stats import genpareto


def compute_return_levels_1d(series):
    quantile = 0.95
    return_periods = np.asarray([20, 50, 100, 200, 500, 1500], dtype="f8")

    valid_series = series[~np.isnan(series)]
    if not len(valid_series):
        return np.full(len(return_periods), np.nan)

    threshold = np.quantile(valid_series, quantile)
    exceedances = valid_series[valid_series > threshold]

    if len(exceedances) < 5:
        # Not enough exceedances for quantile {quantile} in series {series}. Returning NaN.
        return np.full(len(return_periods), np.nan)

    excesses = exceedances - threshold
    shape, loc, scale = genpareto.fit(excesses)

    quantiles = threshold + genpareto.ppf(1 - 1 / return_periods, shape, loc, scale)

    return quantiles


def thornthwaite(T):
    # Ensure temperature is non-negative (Thornthwaite assumes T >= 0)
    T = np.maximum(T, 0)

    # Calculate heat index (I) for each grid point
    # Sum over months for each grid point
    I = (T.sum(dim="time") / 5) ** 1.514

    # Calculate empirical exponent (a) for each grid point
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239

    # Extract month as integer 1-12
    month = T.coords["time"].dt.month
    # Calculate daylight hours correction factor for each month
    daylight_hours = 12 + 4 * np.sin(
        (2 * np.pi * (month - 1) / 12) + (T.coords["lat"] * np.pi / 180)
    )
    # Expand daylight_hours to match the grid shape with lon
    daylight_hours_grid = daylight_hours.expand_dims(lon=T.lon)

    # Compute PET for each grid point
    pet = 16 * (10 * T / I) ** a * daylight_hours_grid / 12

    return pet


def compute_spei(pe, timescale):
    pe_array = pe.sel(lat=pe.coords["lat"][0], lon=pe.coords["lon"][0]).values
    pe_series = pd.Series(data=pe_array, index=pd.DatetimeIndex(data=pe.time))
    spei_series = si.spei(pe_series, timescale=timescale, fit_freq="ME")
    return spei_series.values


# Wrapper to compute return levels per block while preserving spatial dims
def spei_return_period_quantiles(tasmax, pr, timescale):
    # Calculate Thornthwaite evapotranspiration as a function of daily maximum temperature
    eva = thornthwaite(tasmax)

    # Calculate precipitation excess, replace NaN values with 0
    pe = (pr - eva).fillna(0).chunk({"lat": 1, "lon": 1, "time": -1})

    spei = xr.map_blocks(
        compute_spei,
        pe,
        args=[timescale],
        template=pe,
    )

    # Compute return levels for each grid point
    return xr.apply_ufunc(
        compute_return_levels_1d,
        spei,
        timescale,
        input_core_dims=[["time"], []],
        output_core_dims=[["return_period"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"return_period": 6}},
    )


def main(task, work_path):
    lon_bounds = (task.lon_min, task.lon_max)
    lat_bounds = (task.lat_min, task.lat_max)
    start_year, end_year = tuple(map(int, task.epoch.split("-")))
    tf_str = f"{start_year}-{end_year}"

    ds = xr.open_zarr(work_path / "nex-gddp-cmip6.zarr").sel(
        lon=slice(*lon_bounds),
        lat=slice(*lat_bounds),
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
        model=task.model,
        scenario=task.scenario,
    )
    tasmax = ds["tasmax"].compute()
    pr = ds["pr"].compute()
    if tasmax.size == 0 or pr.size == 0:
        logging.warning("Time series is empty. Skipping this task")
        return

    tasmax = tasmax.chunk({"time": -1, "lat": 30, "lon": 30}).compute()
    pr = pr.chunk({"time": -1, "lat": 30, "lon": 30}).compute()

    result3 = spei_return_period_quantiles(tasmax, pr, 90).compute()
    result3 = result3.assign_coords(
        return_period=np.asarray([20, 50, 100, 200, 500, 1500], dtype="f8")
    ).expand_dims(model=[task.model], scenario=[task.scenario], epoch=[tf_str])
    result3.name = "spei3_return_level"
    result3.to_zarr(work_path / "nex-gddp-cmip6.return_levels.zarr", region="auto")

    result12 = spei_return_period_quantiles(tasmax, pr, 360).compute()
    result12 = result12.assign_coords(
        return_period=np.asarray([20, 50, 100, 200, 500, 1500], dtype="f8")
    ).expand_dims(model=[task.model], scenario=[task.scenario], epoch=[tf_str])
    result12.name = "spei12_return_level"
    result12.to_zarr(work_path / "nex-gddp-cmip6.return_levels.zarr", region="auto")
    logging.info(f"Saved quantiles", flush=True)


if __name__ == "__main__":
    try:
        work_path = Path(sys.argv[1])
        job_num = int(sys.argv[2])
        total_num = int(sys.argv[3])
    except IndexError:
        print(f"Usage: python {__file__} <work_path> <start_num> <end_num>")
        sys.exit()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting the script to process climate data...")

    all_tasks = pd.read_csv(work_path / "spei_tasks.csv")
    ntasks = len(all_tasks)

    # Select tasks for this job_num (stride through ntasks with spacing of total_num)
    task_ids = np.arange(job_num, ntasks, total_num)
    tasks = all_tasks.iloc[task_ids].copy()
    variable = "spei"
    task_ids = list(map(int, task_ids))
    logging.info(f"Processing: {variable=}, {len(task_ids)} tasks: {task_ids=}")

    for task in tasks.itertuples():
        main(task, work_path)
