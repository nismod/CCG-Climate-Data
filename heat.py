"""heat.py - Script for Handling Remote Dataset Metadata"""

import datetime
import logging
import sys
import time
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.stats import genpareto


def compute_return_levels_1d(series):
    quantile = 0.95
    time_scale = 1
    return_periods = np.asarray([20, 50, 100, 200, 500, 1500], dtype="f8")
    valid_series = series[~np.isnan(series)]
    if not len(valid_series):
        return np.full(len(return_periods), np.nan)

    threshold = np.quantile(valid_series, quantile)
    exceedances = valid_series[valid_series > threshold]

    if len(exceedances) < 5:
        # Not enough exceedances for quantile {quantile} in series {series}. Returning NaN
        return np.full(len(return_periods), np.nan)

    excesses = exceedances - threshold
    shape, loc, scale = genpareto.fit(excesses)

    quantiles = threshold + genpareto.ppf(
        1 - 1 / (return_periods * time_scale), shape, loc, scale
    )

    return quantiles


def return_period_quantiles(x):
    return xr.apply_ufunc(
        compute_return_levels_1d,
        x,
        input_core_dims=[["time"]],
        output_core_dims=[["return_period"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"return_period": 6}},
    )


# Loop through all models and scenarios
def main(task, file_meta):
    scenario = task.scenario
    model = task.model
    lon_bounds = (task.lon_min, task.lon_max)
    lat_bounds = (task.lat_min, task.lat_max)

    if scenario == "historical":
        time_frames = [[1984, 2014]]
    else:
        time_frames = [[2015, 2040], [2041, 2065], [2066, 2090]]

    for tf in time_frames:
        #
        # Read
        #
        start_year, end_year = tf
        tf_str = f"{start_year}-{end_year}"

        logging.info(
            f"Reading years {tf_str} for Model {model}, Scenario {scenario}, lon {lon_bounds}, lat {lat_bounds}",
        )
        read_start = time.time()

        # Filter files that match the model, scenario and years
        # years = list(map(str, range(start_year, end_year + 1)))

        ds = xr.open_zarr(f"nex-gddp-cmip6.{model}.{scenario}.zarr").sel(
            lon=slice(*lon_bounds),
            lat=slice(*lat_bounds),
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
            model=task.model,
            scenario=task.scenario,
        )
        tasmax = ds["tasmax"].compute()

        read_end = time.time()
        read_execution = datetime.timedelta(seconds=read_end - read_start)
        logging.info(f"Data loading time: {read_execution}")

        #
        # Compute
        #:,
        compute_start = time.time()
        if tasmax.size == 0:
            logging.info(
                "Warning: time_series is empty. Skipping this grid point.",
            )
            compute_end_early = time.time()
            compute_execution_early = datetime.timedelta(
                seconds=compute_end_early - compute_start
            )
            logging.info(
                f"Compute return levels (end early) time: {compute_execution_early} seconds",
            )
            continue

        # Build dask result with shape (len(TR), lat, lon); reduce time only
        return_periods = np.asarray(return_periods, dtype="f8")

        # Ensure single time chunk so we reduce time -> return_period once
        tasmax = tasmax.chunk({"time": -1, "lat": 30, "lon": 30})
        logging.info(f"tasmax chunks: {tasmax.chunks}")

        result = return_period_quantiles(tasmax)

        final_result = result.compute()
        # Debug: Check the shape of final_result
        logging.info(f"Shape of final_result: {final_result.shape}")

        # No manual reshape; final_result is expected (len(TR), lat, lon)
        result_xr = final_result.assign_coords(
            return_period=return_periods
        ).expand_dims(model=[task.model], scenario=[task.scenario], epoch=[tf_str])
        result_xr.name = "tasmax_return_level"

        compute_end = time.time()
        compute_execution = datetime.timedelta(seconds=compute_end - compute_start)
        logging.info(f"Compute return levels time: {compute_execution} seconds")

        #
        # Plot and save
        #
        plot_save_start = time.time()

        # Ensure the quantiles_to_plot is a 2D array for plotting
        result_xr.to_zarr("tasmax_quantiles.zarr", region="auto")
        logging.info(
            f"Saved quantiles for {model=} {scenario=} {tf=}",
        )

        # plot
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mollweide())
        ax.set_extent(
            [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, facecolor="lightblue")
        ax.add_feature(cfeature.RIVERS, edgecolor="blue")
        ax.set_title(
            f"tasmax Quantiles for {model} - {scenario} - {tf_str}", fontsize=16
        )
        # Plot the quantiles for the selected return period
        plot_rp = 50
        quantiles_to_plot = (
            result_xr.sel(
                return_period=plot_rp, model=task.model, scenario=task.scenario
            ).data
            - 273.15
        )  # degrees K to C
        plt.pcolormesh(
            result_xr.lon.data,
            result_xr.lat.data,
            quantiles_to_plot,
            cmap="coolwarm",
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        plt.colorbar(label=f"tasmax RP {plot_rp}", orientation="vertical")
        plt.savefig("plots" / f"tasmax_quantiles_{model}_{scenario}_{tf_str}.png")
        plt.close()  # Close the plot to free memory

        # End the timer
        plot_save_end = time.time()
        plot_save_execution = datetime.timedelta(
            seconds=plot_save_end - plot_save_start
        )
        logging.info(f"Plot and save time: {plot_save_execution} seconds")


if __name__ == "__main__":
    try:
        path_save = Path(sys.argv[1])
        job_num = int(sys.argv[2])
        total_num = int(sys.argv[3])
    except IndexError:
        print(f"Usage: python {__file__} <save_path> <start_num> <end_num>")
        sys.exit()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting the script to process climate data...")

    all_tasks = pd.read_csv(path_save / "tasmax_tasks.csv")
    ntasks = len(all_tasks)

    # Select tasks for this job_num (stride through ntasks with spacing of total_num)
    task_ids = np.arange(job_num, ntasks, total_num)
    tasks = all_tasks.iloc[task_ids].copy()
    variable = "tasmax"
    logging.info(f"Processing variable: {variable}")

    for task in tasks.itertuples():
        main(task, file_meta)
        break
