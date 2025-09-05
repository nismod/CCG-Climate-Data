"""Heat.py - Script for Handling Remote Dataset Metadata

This script is designed to work with remote datasets hosted on an Amazon S3
bucket. It performs the following tasks:

1. Suppresses specific warnings from the PyDAP library to ensure clean output.
2. Defines a list of URLs pointing to metadata or checksum files for datasets.

The URLs provided in the script can be used to access and process metadata for
climate-related datasets, such as those from the CMIP6 project.

Dependencies:

- requests: For making HTTP requests to fetch data.
"""

import multiprocessing as mp
import os
import sys
import time
import warnings
from functools import partial

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
import dask.array as da
import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
import s3fs
import xarray as xr
from dask.array import map_blocks
from netCDF4 import Dataset
from scipy.stats import genpareto


try:
    path_save = str(sys.argv[1])
    job_num = int(sys.argv[2])
    total_num = int(sys.argv[3])
except IndexError:
    print(f"Usage: python {__file__} <save_path> <start_num> <end_num>")
    sys.exit()

# Suppress PyDAP warnings about DAP protocol
warnings.filterwarnings(
    "ignore", message="PyDAP was unable to determine the DAP protocol"
)

# Setup S3 file system
fs = s3fs.S3FileSystem(anon=True)

# URLs of the text files
urls = ["https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v2.0_md5.txt"]

# Lists to store extracted information
tasmax_files = []  # Stores full paths of files containing 'tasmax'

models = set()  # Unique models (first element after root)
scenarios = set()  # Unique scenarios (second element after root)
variables = set()  # Unique variables (fourth element after root)
years = set()

# Process each file
for url in urls:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if request fails

        # Process each line in the file
        for line in response.text.splitlines():
            parts = line.split()  # Split line into columns
            if len(parts) == 2:  # Ensure valid format
                md5_hash, file_path = parts
                if "tasmax" in file_path:
                    tasmax_files.append(file_path)

                # Extract model, scenario, and variable from file path
                path_parts = file_path.split("/")
                if len(path_parts) >= 5:
                    models.add(path_parts[1])
                    scenarios.add(path_parts[2])
                    variables.add(path_parts[4])
                    year = (
                        path_parts[5].split("_")[-1].split(".")[0]
                    )  # Extract the year
                    years.add(year)  # Add the year to the set

    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}", flush=True)

# Convert sets to lists
models = sorted(models)
scenarios = sorted(scenarios)
variables = sorted(variables)

num_time_frames = 4
num_TR_computed = 6

# Base URL for OpenDAP server
base_url = "s3://nex-gddp-cmip6/"

# Define the return periods to calculate
TR = [20, 50, 100, 200, 500, 1500]
return_periods_empirical = [5, 10, 15, 20, 25, 30]
return_periods_theoretical = np.linspace(2, 2000, num=1000)

lat_cells = 600  # Number of latitude cells
lon_cells = 1440  # Number of longitude cells
# Initialize arrays with NaN values
tasmax_quantiles = np.full(
    (
        lat_cells,
        lon_cells,
        len(scenarios),
        num_time_frames,
        len(models),
        num_TR_computed,
    ),
    np.nan,
)

grid_div = 16  # Number of grid divisions


vector_indices = np.arange(job_num, len(models) * len(scenarios) * grid_div, total_num)
indices_grid = np.floor(vector_indices / (len(models) * len(scenarios)))
indices_models = np.floor(
    (vector_indices - indices_grid * (len(models) * len(scenarios))) / (len(scenarios))
)
indices_scenarios = np.floor(
    (
        vector_indices
        - indices_models * (len(models) * len(scenarios))
        - indices_models * (len(scenarios))
    )
)

# Calculate the grid size (sqrt of total_number)
grid_size = int(np.sqrt(grid_div))  # Number of divisions along each axis

# Define the latitude and longitude ranges
lat_range = (-60, 90)  # Latitude range
lon_range = (0, 360)  # Longitude range

# Divide the latitude and longitude into equal intervals
lat_intervals = np.linspace(lat_range[0], lat_range[1], grid_size + 1)
lon_intervals = np.linspace(lon_range[0], lon_range[1], grid_size + 1)

# Determine the grid cell corresponding to the job_number
lat_idx = (indices_grid // grid_size).astype(int)  # Row index in the latitude grid
lon_idx = (indices_grid % grid_size).astype(int)  # Column index in the longitude grid

# Get the latitude and longitude bounds for the specific grid cell
lat_bounds = (lat_intervals[lat_idx], lat_intervals[lat_idx + 1])
lon_bounds = (lon_intervals[lon_idx], lon_intervals[lon_idx + 1])

models = models[indices_models]  # Limit to the specified number of models
scenarios = scenarios[indices_scenarios]  # Limit to the specified number of scenarios
print("Starting the script to process climate data...", flush=True)


def compute_return_levels_1d(series, quantile, return_periods, time_scale):
    series = np.asarray(series)
    valid_series = series[~np.isnan(series)]
    threshold = np.quantile(valid_series, quantile)
    exceedances = valid_series[valid_series > threshold]

    if len(exceedances) < 5:
        print(
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


variable = "tasmax"

print(f"Processing variable: {variable}", flush=True)


def filter_files(files, model, scenario, ensemble_member, variable, year_strings):
    return [
        file
        for file in files
        if model in file
        and scenario in file
        and ensemble_member in file
        and variable in file
        and any(year in file for year in year_strings)
    ]


# Wrapper to compute return levels per block while preserving spatial dims
def _return_levels_block(block, quantile, tr, time_scale=1):
    if block.size == 0:
        print("Warning: Empty block encountered. Skipping.", flush=True)
        return np.full((len(tr),) + block.shape[1:], np.nan)

    # print(f"Processing block with shape: {block.shape}", flush=True)
    # print(f"TR: {tr}, Time scale: {time_scale}")

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
    # print(f"Finished processing block : {out}", flush = True)
    return out


# Loop through all models and scenarios
for model in models:
    for scenario in scenarios:
        if scenario == "historical":
            time_frames = [[1984, 2014]]
            ttf = 0
        else:
            time_frames = [[2015, 2040], [2041, 2065], [2066, 2090]]
            ttf = [1, 2, 3]

        for tf_idx, tf in enumerate(time_frames):
            start_time = time.time()
            # yearly_data = []
            # Filter files that match the model, scenario, and end with '_v2.0.nc'
            matching_files = [
                file
                for file in tasmax_files
                if model in file and scenario in file and file.endswith("_v2.0.nc")
            ]
            print(
                f"Reading year {tf} for Model {model}, Scenario {scenario}", flush=True
            )
            if not matching_files:
                print(
                    f"Skipping Model {model}, Scenario {scenario}, no matching files ending with '_v2.0.nc' found.",
                    flush=True,
                )
                continue

            # Open the dataset
            filenames = []
            # Restrict files to those containing years within the range specified by tf
            filenames = [
                f"{base_url}{file}"
                for file in matching_files
                if any(str(year) in file for year in range(tf[0], tf[1] + 1))
            ]

            def _preprocess(x, lon_bounds, lat_bounds):
                return x.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

            filesystem = fsspec.filesystem("s3", anon=True)
            filehandles = [filesystem.open(filename) for filename in filenames]
            preprocessor = partial(
                _preprocess, lon_bounds=lon_bounds, lat_bounds=lat_bounds
            )
            ds = xr.open_mfdataset(
                filehandles,
                preprocess=preprocessor,
                parallel=True,
                engine="h5netcdf",
                chunks={"time": 100},
            )
            tasmax = ds["tasmax"]
            # print('Sample of tasmax data:', tasmax.isel(time=1000, lat=2, lon=2).values, flush = True)
            lat_coords = ds["lat"].values
            lon_coords = ds["lon"].values
            # print(f"tasmax shape: {tasmax.shape}", flush = True)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.4f} seconds", flush=True)
            start_time = time.time()
            if tasmax.size == 0:
                print(
                    "Warning: time_series is empty. Skipping this grid point.",
                    flush=True,
                )
                break

            # Build dask result with shape (len(TR), lat, lon); reduce time only
            TR = np.asarray(TR, dtype="f8")
            # Ensure single time chunk so we reduce time -> return_period once
            # Check for NaN values in tasmax
            if np.isnan(tasmax).all():
                print(
                    "Warning: All values in tasmax are NaN. Skipping computation.",
                    flush=True,
                )
                break

            # Ensure proper chunking
            tasmax = tasmax.chunk({"time": -1, "lat": 1, "lon": 1})
            print(f"tasmax chunks: {tasmax.chunks}", flush=True)

            # Validate TR
            if len(TR) == 0:
                raise ValueError(
                    "TR (return periods) is empty. Cannot compute quantiles."
                )

            result = da.map_blocks(
                _return_levels_block,  # expects (time, y, x) -> (len(TR), y, x)
                tasmax.data,
                quantile=0.95,  # dask array (time, lat, lon)
                tr=TR,  # return periods (broadcasted arg)
                time_scale=1,
                dtype="f8",
                chunks=((len(TR),),)
                + tuple(tasmax.data.chunks[1:]),  # (len(TR), lat_chunks, lon_chunks)
            )

            final_result = result.compute()
            # Debug: Check the shape of final_result
            print(f"Shape of final_result: {final_result.shape}", flush=True)

            # No manual reshape; final_result is expected (len(TR), lat, lon)
            result_xr = xr.DataArray(
                final_result,
                dims=["return_period", "lat", "lon"],
                coords={"return_period": TR, "lat": lat_coords, "lon": lon_coords},
                name="tasmax_return_level",
            )

            print(
                "Computed quantiles for Model {}, Scenario {}, Time Frame {}.".format(
                    model, scenario, tf
                ),
                flush=True,
            )
            print(result_xr[:, 2, 2], flush=True)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.4f} seconds", flush=True)
            start_time = time.time()

            # Define folder structure: path_save/Scenario/Model/Time_Frame/
            scenario_dir = os.path.join(path_save, scenario)
            model_dir = os.path.join(scenario_dir, model)
            time_frame_dir = os.path.join(model_dir, str(tf))
            os.makedirs(time_frame_dir, exist_ok=True)
            output_folder = os.path.join(path_save, scenario, model, str(tf) + "/")
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            # Calculate quantiles for each return period
            quantiles_to_plot = result_xr.values[:, :, :]

            # plot
            plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
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
                f"tasmax Quantiles for {model} - {scenario} - {tf}", fontsize=16
            )
            # Plot the quantiles for the first return period
            quantiles_to_plot = quantiles_to_plot[
                3, :, :
            ]  # Select the first return period for plotting
            plt.pcolormesh(
                lon_coords,
                lat_coords,
                quantiles_to_plot,
                cmap="coolwarm",
                shading="auto",
                transform=ccrs.PlateCarree(),
            )
            plt.colorbar(label="tasmax Quantiles", orientation="vertical")
            plt.savefig(f"{output_folder}tasmax_quantiles_{model}_{scenario}_{tf}.png")
            plt.close()  # Close the plot to free memory

            # Ensure the quantiles_to_plot is a 2D array for plotting
            result_xr.to_netcdf(
                os.path.join(
                    output_folder,
                    f"tasmax_quantiles_{model}_{scenario}_{tf}_n_{job_num}.nc",
                )
            )
            print(
                f"Saved quantiles to {output_folder}tasmax_quantiles_{model}_{scenario}_{tf}_n_{job_num}.nc",
                flush=True,
            )


# End the timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
