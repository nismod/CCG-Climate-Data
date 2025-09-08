"""heat.py - Script for Handling Remote Dataset Metadata"""

import datetime
import logging
import sys
import time
from functools import partial
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.stats import genpareto


def download_file(url, fname):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def read_meta(path_save, nsubgrids):
    """Download (if necessary) and read metadata, build task list of model/scenario/subgrid combinations"""
    meta_url = "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v2.0_md5.txt"
    meta_fname = path_save / "index_v2.0_md5.txt"

    if not meta_fname.exists():
        download_file(meta_url, meta_fname)

    meta = pd.read_csv(
        meta_fname, sep=" ", header=None, names=["hash", "_", "path"]
    ).drop(columns="_")
    meta = meta[meta.path.str.contains("tasmax")]
    meta = meta[meta.path.str.endswith("_v2.0.nc")]

    meta[["model", "scenario", "variable", "year"]] = meta.path.str.extract(
        r"[^/]+/"
        r"(?P<model>[^/]+)/"
        r"(?P<scenario>[^/]+)/"
        r"[^/]+/"
        r"(?P<variable>[^/]+)/"
        r".*"
        r"(?P<year>\d\d\d\d)"
        r".*"
    )

    model_scenario_combinations = (
        meta[["model", "scenario"]]
        .drop_duplicates()
        .sort_values(by=["model", "scenario"])
        .reset_index(drop=True)
    )
    all_task_dfs = []
    for grid_idx in range(nsubgrids):
        task_df = model_scenario_combinations.copy()
        task_df["grid_idx"] = grid_idx
        all_task_dfs.append(task_df)

    return pd.concat(all_task_dfs).reset_index(drop=True), meta


def add_bounds(tasks, grid_size):
    # Define the latitude and longitude ranges
    lat_range = (-60, 90)  # Latitude range
    lon_range = (0, 360)  # Longitude range

    # Divide the latitude and longitude into equal intervals
    lat_intervals = np.linspace(lat_range[0], lat_range[1], grid_size + 1)
    lon_intervals = np.linspace(lon_range[0], lon_range[1], grid_size + 1)

    # Determine the grid cell corresponding to the job_number
    indices_grid = tasks.grid_idx
    lat_idx = (indices_grid // grid_size).astype(int)
    lon_idx = (indices_grid % grid_size).astype(int)

    # Get the latitude and longitude bounds for the specific grid cell
    lat_bounds = (lat_intervals[lat_idx], lat_intervals[lat_idx + 1])
    lon_bounds = (lon_intervals[lon_idx], lon_intervals[lon_idx + 1])

    tasks["lat_min"] = lat_bounds[0]
    tasks["lat_max"] = lat_bounds[1]
    tasks["lon_min"] = lon_bounds[0]
    tasks["lon_max"] = lon_bounds[1]
    return tasks


def compute_return_levels_1d(series, quantile, return_periods, time_scale):
    series = np.asarray(series)
    valid_series = series[~np.isnan(series)]
    threshold = np.quantile(valid_series, quantile)
    exceedances = valid_series[valid_series > threshold]

    if len(exceedances) < 5:
        logging.info(
            f"Not enough exceedances for quantile {quantile} in series {series}. Returning NaN.",
            flush=True,
        )
        return np.full(len(return_periods), np.nan)

    excesses = exceedances - threshold
    shape, loc, scale = genpareto.fit(excesses)

    quantiles = [
        threshold + genpareto.ppf(1 - 1 / (t * time_scale), shape, loc, scale)
        for t in return_periods
    ]

    return np.array(quantiles)


# Wrapper to compute return levels per block while preserving spatial dims
def _return_levels_block(block, quantile, tr, time_scale=1):
    if block.size == 0:
        logging.warning("Empty block encountered. Skipping.", flush=True)
        return np.full((len(tr),) + block.shape[1:], np.nan)

    logging.debug(f"Processing block with shape: {block.shape}", flush=True)
    logging.debug(f"TR: {tr}, Time scale: {time_scale}")

    block = np.asarray(block)
    t, ny, nx = block.shape
    out = np.full((len(tr), ny, nx), np.nan, dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            ts = block[:, j, i]
            if np.all(np.isnan(ts)):
                continue
            out[:, j, i] = compute_return_levels_1d(
                ts, quantile, tr, time_scale=time_scale
            )
    logging.debug(f"Finished processing block : {out}", flush=True)
    return out


# Loop through all models and scenarios
def main(task, file_meta, return_periods):
    # Validate
    if len(return_periods) == 0:
        raise ValueError("TR (return periods) is empty. Cannot compute quantiles.")

    scenario = task.scenario
    model = task.model
    # lon_bounds = (task.lon_min, task.lon_max)
    # lat_bounds = (task.lat_min, task.lat_max)
    lon_bounds = (0.0, 0.5)
    lat_bounds = (51.0, 51.5)

    if scenario == "historical":
        time_frames = [[1984, 2014]]
    else:
        time_frames = [[2015, 2040], [2041, 2065], [2066, 2090]]

    for tf in time_frames:
        #
        # Read
        #
        logging.info(
            f"Reading years {tf} for Model {model}, Scenario {scenario}, lon {lon_bounds}, lat {lat_bounds}",
            flush=True,
        )
        start_year, end_year = tf
        read_start = time.time()

        # Filter files that match the model, scenario and years
        years = list(map(str, range(start_year, end_year + 1)))
        matching_files = file_meta.query(
            f"scenario == '{scenario}' and model == '{model}' and year in {years}"
        )
        if matching_files.empty:
            logging.info(
                f"Skipping Model {model}, Scenario {scenario}, no matching files ending with '_v2.0.nc' found.",
                flush=True,
            )
            continue

        filenames = matching_files.path
        filesystem = fsspec.filesystem("file")

        def _read_local(fn, filesystem):
            return filesystem.open(Path("./data") / Path(fn).name)

        filehandles = [_read_local(fn, filesystem) for fn in filenames]

        def _preprocess(x, lon_bounds, lat_bounds):
            return x.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

        preprocessor = partial(
            _preprocess, lon_bounds=lon_bounds, lat_bounds=lat_bounds
        )
        ds = xr.open_mfdataset(filehandles, preprocess=preprocessor, parallel=True)
        tasmax = ds["tasmax"]
        lat_coords = ds["lat"].values
        lon_coords = ds["lon"].values
        read_end = time.time()
        read_execution = datetime.timedelta(seconds=read_end - read_start)
        logging.info(f"Data loading time: {read_execution}", flush=True)

        #
        # Compute
        #
        compute_start = time.time()
        if tasmax.size == 0:
            logging.info(
                "Warning: time_series is empty. Skipping this grid point.",
                flush=True,
            )
            compute_end_early = time.time()
            compute_execution_early = datetime.timedelta(
                seconds=compute_end_early - compute_start
            )
            logging.info(
                f"Compute return levels (end early) time: {compute_execution_early} seconds",
                flush=True,
            )
            continue

        # Build dask result with shape (len(TR), lat, lon); reduce time only
        return_periods = np.asarray(return_periods, dtype="f8")

        # Ensure single time chunk so we reduce time -> return_period once
        tasmax = tasmax.chunk({"time": -1, "lat": 30, "lon": 30})
        logging.info(f"tasmax chunks: {tasmax.chunks}", flush=True)

        result = da.map_blocks(
            _return_levels_block,  # expects (time, y, x) -> (len(TR), y, x)
            tasmax.data,
            # dask array (time, lat, lon)
            quantile=0.95,
            # return periods (broadcasted arg)
            tr=return_periods,
            time_scale=1,
            dtype="f8",
            # (len(TR), lat_chunks, lon_chunks)
            chunks=((len(return_periods),),) + tuple(tasmax.data.chunks[1:]),
        )

        final_result = result.compute()
        # Debug: Check the shape of final_result
        logging.info(f"Shape of final_result: {final_result.shape}", flush=True)

        # No manual reshape; final_result is expected (len(TR), lat, lon)
        result_xr = xr.DataArray(
            final_result,
            dims=["return_period", "lat", "lon"],
            coords={
                "return_period": return_periods,
                "lat": lat_coords,
                "lon": lon_coords,
            },
            name="tasmax_return_level",
        )

        compute_end = time.time()
        compute_execution = datetime.timedelta(seconds=compute_end - compute_start)
        logging.info(
            f"Compute return levels time: {compute_execution} seconds", flush=True
        )

        #
        # Plot and save
        #
        plot_save_start = time.time()

        # Define folder structure: path_save/Scenario/Model/Time_Frame/
        output_folder = path_save / scenario / model / f"{start_year}-{end_year}"
        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        # Calculate quantiles for each return period
        quantiles_to_plot = result_xr.values[:, :, :]

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
        ax.set_title(f"tasmax Quantiles for {model} - {scenario} - {tf}", fontsize=16)
        # Plot the quantiles for the selected return period
        quantiles_to_plot = quantiles_to_plot[3, :, :]
        plt.pcolormesh(
            lon_coords,
            lat_coords,
            quantiles_to_plot,
            cmap="coolwarm",
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        plt.colorbar(label="tasmax Quantiles", orientation="vertical")
        plt.savefig(output_folder / "tasmax_quantiles_{model}_{scenario}_{tf}.png")
        plt.close()  # Close the plot to free memory

        # Ensure the quantiles_to_plot is a 2D array for plotting
        result_xr.to_netcdf(
            output_folder / f"tasmax_quantiles_{model}_{scenario}_{tf}_n_{job_num}.nc"
        )
        logging.info(
            f"Saved quantiles to {output_folder}/tasmax_quantiles_{model}_{scenario}_{tf}_n_{job_num}.nc",
            flush=True,
        )
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

    # Define the return periods to calculate
    TR = [20, 50, 100, 200, 500, 1500]

    lat_cells = 600  # Number of latitude cells
    lon_cells = 1440  # Number of longitude cells
    grid_size = 4  # Number of divisions along each axis
    grid_div = grid_size**2  # Number of grid divisions

    logging.info("Starting the script to process climate data...", flush=True)

    all_tasks, file_meta = read_meta(path_save, grid_div)
    ntasks = len(all_tasks)

    # Select tasks for this job_num (stride through ntasks with spacing of total_num)
    task_ids = np.arange(job_num, ntasks, total_num)
    tasks = all_tasks.iloc[task_ids].copy()
    tasks = add_bounds(tasks, grid_size)

    variable = "tasmax"
    logging.info(f"Processing variable: {variable}", flush=True)

    for task in tasks.itertuples():
        main(task, file_meta, TR)
        break
