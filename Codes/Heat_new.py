import warnings
import pydap.client
import requests
import sys
import numpy as np
from scipy.stats import genpareto
import multiprocessing as mp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import xarray as xr
import s3fs
from netCDF4 import Dataset
import h5py
import fsspec
import time
from scipy.stats import genpareto
import fsspec
from functools import partial

"""
Heat.py - Script for Handling Remote Dataset Metadata

This script is designed to work with remote datasets hosted on an Amazon S3 bucket.
It performs the following tasks:
1. Suppresses specific warnings from the PyDAP library to ensure clean output.
2. Defines a list of URLs pointing to metadata or checksum files for datasets.

The URLs provided in the script can be used to access and process metadata for 
climate-related datasets, such as those from the CMIP6 project.

Dependencies:
- warnings: For managing and suppressing warnings.
- pydap.client: For accessing data using the OPeNDAP protocol.
- requests: For making HTTP requests to fetch data.
- sys: For system-specific operations (not yet used in this snippet).

"""
path_save = "/soge-home/cenv0972/CCG/Heat/CMIP6/"  # Path to save the results

# Suppress PyDAP warnings about DAP protocol
warnings.filterwarnings("ignore", message="PyDAP was unable to determine the DAP protocol")

# Setup S3 file system
fs = s3fs.S3FileSystem(anon=True)

# URLs of the text files
urls = ["https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v2.0_md5.txt"]

# Lists to store extracted information
tasmax_files = []  # Stores full paths of files containing 'tasmax'

models = set()  # Unique models (first element after root)
scenarios = set()  # Unique scenarios (second element after root)
variables = set()  # Unique variables (fourth element after root)
years=set()

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
                    year = path_parts[5].split("_")[-1].split(".")[0]  # Extract the year
                    years.add(year)  # Add the year to the set
                    
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}", flush = True)

# Convert sets to lists
models = sorted(models)
scenarios = sorted(scenarios)
variables = sorted(variables)

num_time_frames=4
num_TR_computed=6

# Base URL for OpenDAP server
base_url = "s3://nex-gddp-cmip6/"
ensemble_member = "r1i1p1f1"  

# Define the return periods to calculate
TR = [20, 50, 100, 200, 500, 1500]
return_periods_empirical = [5, 10, 15, 20, 25, 30]      
return_periods_theoretical= np.linspace(2,2000,num=1000) 

# Define the output folder path
output_folder = "/CCG/Heat/"

lat_cells = 600  # Number of latitude cells
lon_cells = 1440  # Number of longitude cells
# Initialize arrays with NaN values
tasmax_quantiles = np.full((lat_cells, lon_cells, len(scenarios), num_time_frames,  len(models), num_TR_computed), np.nan)

# try:
    # job_num = int(sys.argv[1])
    # total_num = int(sys.argv[2])
# except IndexError:
    # print(f"Usage: python {__file__} <start_num> <end_num>")
    # sys.exit()
total_num = 10000
job_num = 0  # Example job number, replace with actual job number if needed
# Calculate the grid size (sqrt of total_number)
grid_size = int(np.sqrt(total_num))  # Number of divisions along each axis

# Define the latitude and longitude ranges
lat_range = (-60, 90)  # Latitude range
lon_range = (0, 360)  # Longitude range

# Divide the latitude and longitude into equal intervals
lat_intervals = np.linspace(lat_range[0], lat_range[1], grid_size + 1)
lon_intervals = np.linspace(lon_range[0], lon_range[1], grid_size + 1)

# Determine the grid cell corresponding to the job_number
lat_idx = job_num // grid_size  # Row index in the latitude grid
lon_idx = job_num % grid_size   # Column index in the longitude grid

# Get the latitude and longitude bounds for the specific grid cell
lat_bounds = (lat_intervals[lat_idx], lat_intervals[lat_idx + 1])
lon_bounds = (lon_intervals[lon_idx], lon_intervals[lon_idx + 1])
#indices = np.arange(job_num)
#models = [models[i] for i in indices]  # Fix model selection

models = models[:1]  # Limit to the specified number of models
scenarios = scenarios[:1]  # Limit to the specified number of scenarios
print('Starting the script to process climate data...', flush = True)

def fit_gpd_clustered_3d(data, quantile, return_periods_computed, time_scale):
    lat_size, lon_size, time_size = data.shape
    results = {}

    for i in range(lat_size):
        for j in range(lon_size):
            time_series = data[i,j,:]
            threshold = np.quantile(time_series, quantile)
            exceedances = time_series[time_series > threshold]
            if len(exceedances) < 5:
                results[(i, j)] = {
                    "Quantiles": np.full(len(return_periods_computed), np.nan)
                }
                continue
            excesses = exceedances - threshold
            shape, loc, scale = genpareto.fit(excesses)
            quantiles = [
                threshold + genpareto.ppf(1 - 1 / (t * time_scale), shape, loc, scale) for t in return_periods_computed
            ]

            results[(i, j)] = {
                "gpd_parameters": {"shape": shape, "scale": scale, "location": loc},
                "Quantiles": quantiles
            }

    return results

variable = "tasmax"  

print(f"Processing variable: {variable}", flush = True)
start_time = time.time() 

# Loop through all models and scenarios
for model in models:
    for scenario in scenarios:
        if scenario == 'historical':
            time_frames=[[1984, 2014]]
            ttf=0
        else:
            time_frames = [[2015, 2040], [2041, 2065], [2066,2090]]
            ttf=[1,2,3]

        for tf_idx, tf in enumerate(time_frames):
            yearly_data = []  
            # Filter files that match the model, scenario, and end with '_v2.0.nc'
            matching_files = [
                file for file in tasmax_files
                if model in file and scenario in file and file.endswith("_v2.0.nc")
            ]
            
            if not matching_files:
                print(f"Skipping Model {model}, Scenario {scenario}, no matching files ending with '_v2.0.nc' found.", flush=True)
                continue
            
            # Open the dataset using the pyDAP client
            filenames = []
            # Restrict files to those containing years within the range specified by tf
            filenames = [f"{base_url}{file}"
                for file in matching_files
                if any(str(year) in file for year in range(tf[0], tf[1] + 1))
            ]

            def _preprocess(x, lon_bounds, lat_bounds):
                return x.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

            filesystem = fsspec.filesystem("s3", anon=True) 
            filehandles = [filesystem.open(filename) for filename in filenames]
            preprocessor = partial(_preprocess, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
            ds = xr.open_mfdataset(filehandles, preprocess=preprocessor, parallel=True, engine="h5netcdf")
            # subset = ds.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
            tasmax = ds['tasmax']
            if tasmax.size == 0:
                print("Warning: time_series is empty. Skipping this grid point.")
                break
            yearly_data.append(tasmax) 
        
            if not yearly_data:
                print(f"No data available for Model {model}, Scenario {scenario}, Time Frame {tf}", flush = True)
                continue  # Skip if no data is available
                
            tasmax_3d = np.concatenate(yearly_data, axis=0)
            print(f"tasmax_3d shape: {tasmax_3d.shape}", flush = True)
            sys.exit()
            
            # Fit GPD and compute quantiles 
            results=fit_gpd_clustered_3d(
                data=tasmax_3d,
                quantile=0.95,
                return_periods_computed=TR,
                time_scale=1
            )
            print(f"Results for Model {model}, Scenario {scenario}, Time Frame {tf} computed successfully.", flush = True)

            scenario_idx = scenarios.index(scenario)
            model_idx = models.index(model)

            try:
                for (i, j), res in results.items():
                    if isinstance(ttf, (list, tuple)):
                        tasmax_quantiles[i, j, scenario_idx, ttf[tf_idx], model_idx, :len(res["Quantiles"])] = res["Quantiles"]
                    elif isinstance(ttf, int):
                        tasmax_quantiles[i, j, scenario_idx, ttf, model_idx, :len(res["Quantiles"])] = res["Quantiles"]
                    else:
                        raise TypeError(f"Unexpected type for ttf: {type(ttf)}. Expected int, list, or tuple.")                    
            except Exception as e:
                print(f"Error: {e}", flush = True)
            print('Read Heat', flush = True)

            # Define folder structure: path_save/Scenario/Model/Time_Frame/
            scenario_dir = os.path.join(path_save, scenario)
            model_dir = os.path.join(scenario_dir, model)
            time_frame_dir = os.path.join(model_dir, str(tf))
            os.makedirs(time_frame_dir, exist_ok=True) 

            # Define the NetCDF file name and save path
            filename = f"tmax_s{scenario}_m{model}_tf{tf}.nc"
            save_path = os.path.join(time_frame_dir, filename)

            # Create an xarray Dataset for saving the SPEI data
            ds = xr.Dataset(
                {
                    "tmax_quantiles": (["i", "j", "return_period"], tasmax_quantiles[:, :, scenarios.index(scenario), tf_idx, :, models.index(model)])
                    },
                coords={
                    "i": range(tasmax_quantiles.shape[0]),  
                    "j": range(tasmax_quantiles.shape[1]), 
                    "return_period": TR,      
                },
                attrs={
                    "description": "Maximum daily temperature",
                    "scenario": scenario,
                    "model": model,
                    "time_frame": tf,
                },
            )

            # Save the Dataset to a NetCDF file
            ds.to_netcdf(save_path)
            print(f"Saved tasmax data to: {save_path}")
            

# End the timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")