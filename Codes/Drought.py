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
from datetime import datetime, timedelta
import numpy as np
import spei as si  # si for standardized index


# Suppress PyDAP warnings about DAP protocol
warnings.filterwarnings("ignore", message="PyDAP was unable to determine the DAP protocol")

# Setup S3 file system
fs = s3fs.S3FileSystem(anon=True)

# URLs of the text files
urls = [
    "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_md5.txt",
    "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v1.1_md5.txt",
    "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v1.2_md5.txt"
]
path_save = "/soge-home/cenv0972/CCG/Drought/CMIP6/"  # Path to save the results

# Lists to store extracted information
tasmax_files = []  # Stores full paths of files containing 'tasmax'
tasmin_files = []  # Stores full paths of files containing 'tasmin'
pr_files = []  # Stores full paths of files containing 'pr'
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
                if "tasmin" in file_path:
                    tasmin_files.append(file_path)
                if "pr" in file_path:
                    pr_files.append(file_path)

                # Extract model, scenario, and variable from file path
                path_parts = file_path.split("/")
                if len(path_parts) >= 5:  
                    models.add(path_parts[1])  
                    scenarios.add(path_parts[2])  
                    variables.add(path_parts[4])  
                    
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}", flush = True)



# Convert sets to lists
models = sorted(models)
scenarios = sorted(scenarios)
variables = sorted(variables)

lat_cells=600 
lon_cells=1440
num_time_frames=4
num_TR_computed=6

# Base URL for OpenDAP server
# base_url = "https://ds.nccs.nasa.gov/thredds/dodsC/AMES/NEX/GDDP-CMIP6/"
base_url = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6/"
# base_url = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6/"
ensemble_member = "r1i1p1f1"  

# Define the return periods to calculate
TR = [20, 50, 100, 200, 500, 1500]
return_periods_empirical = [5, 10, 15, 20, 25, 30]      
return_periods_theoretical= np.linspace(2,2000,num=1000) 

# Initialize arrays with NaN values
SPEI_3_quantiles = np.full((lat_cells, lon_cells, len(scenarios), num_time_frames, len(models), num_TR_computed), np.nan)
SPEI_12_quantiles = np.full((lat_cells, lon_cells, len(scenarios), num_time_frames, len(models), num_TR_computed), np.nan)

#try:
#    job_num = int(sys.argv[1])
#    total_num = int(sys.argv[2])
#except IndexError:
#    print(f"Usage: python {__file__} <start_num> <end_num>")
#    sys.exit()

#indices = np.arange(job_num)
#models = [models[i] for i in indices]  # Fix model selection
models = models[:1]  # Limit to the specified number of models
scenarios = scenarios[:1]  # Limit to the specified number of scenarios
print('Starting the script to process climate data...', flush = True)

def fit_gpd_clustered_3d(data, quantile, return_periods_computed, time_scale):
    """
    Fits GPD to clusters of exceedances above a threshold along the time axis (years * days)
    for each (lat, lon) grid point.
    
    Parameters:
    - data: 3D NumPy array (lat, lon, time)
    - quantile: Threshold quantile for exceedances
    - time_scale: Time scale (e.g., number of years)
    
    Returns:
    - Dictionary containing fitted GPD parameters and quantiles for each (lat, lon) point.
    """
    
    lat_size, lon_size, time_size = data.shape
    results = {}

    for i in range(lat_size):
        for j in range(lon_size):
            # Extract time series for each (lat, lon) point
            time_series = data[:,i,j]

            # Compute threshold
            threshold = np.quantile(time_series, quantile)

            # Identify exceedances
            exceedances = time_series[time_series > threshold]

            # Ensure there are at least 5 exceedances for fitting
            if len(exceedances) < 5:
                results[(i, j)] = {
                    "Quantiles": np.full(len(return_periods_computed), np.nan)
                }
                continue

            # Fit GPD to excesses
            excesses = exceedances - threshold
            shape, loc, scale = genpareto.fit(excesses)

            # Compute quantiles for computed ratios
            quantiles = [
                threshold + genpareto.ppf(1 - 1 / (t * time_scale), shape, loc, scale) for t in return_periods_computed
            ]

            # Store results for this grid point
            results[(i, j)] = {
                "gpd_parameters": {"shape": shape, "scale": scale, "location": loc},
                "Quantiles": quantiles
            }

    return results
start_time = time.time()  # Start the timer

import numpy as np

def thornthwaite(lat, T, time):
    """
    Compute potential evapotranspiration (PET) using Thornthwaite's equation.
    
    Parameters:
        lat (float): Latitude in degrees.
        T (np.ndarray): Mean monthly temperatures (°C) with shape (months, latitudes, longitudes).
        time (np.ndarray): Time vector as a NumPy array of datetime64 objects.
    
    Returns:
        np.ndarray: PET values with shape (months, latitudes, longitudes).
    """
    # Ensure temperature is non-negative (Thornthwaite assumes T >= 0)
    T = np.maximum(T, 0)
    
    # Calculate heat index (I) for each grid point
    I = np.sum((T / 5) ** 1.514, axis=0)  # Sum over months for each grid point
    if np.any(I == 0):
        raise ValueError("Heat index (I) is zero for some grid points. Cannot compute PET.")
    
    # Calculate empirical exponent (a) for each grid point
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    
    # Calculate daylight hours correction factor for each month
    daylight_hours = []
    for date in time.astype('datetime64[D]'):
        month = int(str(date)[5:7])  # Extract the month from the datetime64 object
        daylight_hours.append(12 + 4 * np.sin((2 * np.pi * (month - 1) / 12) + (lat * np.pi / 180)))
    daylight_hours = np.array(daylight_hours)  # Convert to NumPy array
    
    # Expand daylight_hours to match the grid shape
    daylight_hours_grid = np.expand_dims(daylight_hours, axis=(1, 2))  # Shape: (months, 1, 1)
    daylight_hours_grid = np.tile(daylight_hours_grid, (1, T.shape[1], T.shape[2]))  # Match grid shape
    
    # Compute PET for each grid point
    PET = 16 * (10 * T / I[np.newaxis, :, :])**a[np.newaxis, :, :] * daylight_hours_grid / 12
    
    return PET

print(f"Processing variables", flush = True)

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
            yearly_data_tasmax = []  
            yearly_data_pr = []
            yearly_data_t = []
            for year in range(tf[0], tf[1] + 1): 
                matching_file = next((file for file in tasmax_files if model in file and scenario in file and str(year) in file), None)
                
                if matching_file:
                    print(f"Found file: {matching_file}", flush = True)
                else:
                    print(f"Skipping Model {model}, Scenario {scenario}, Year {year}, not found in repository.", flush = True)
                    continue
            
                print(f"\nAttempting to open: Model {model}, Scenario {scenario}, Year {year}", flush = True)
                
            
                # Open the dataset using the pyDAP client
                try:
                    s3_url = f"{base_url}{model}/{scenario}/r1i1p1f1/tasmax/tasmax_day_{model}_{scenario}_r1i1p1f1_gn_{year}.nc"
                    fs = fsspec.filesystem("https")
                    ds = xr.open_dataset(s3_url, engine="h5netcdf")
                    lat = ds['lat'].values  # Extract latitudes as a numpy array
                    lon = ds['lon'].values  # Extract longitudes as a numpy array
                    t = ds['time'].values  # Extract time as a numpy array
                    lat_idx = 10   # example index in the latitude array
                    lon_idx = 25   # example index in the longitude array
                    tasmax = ds['tasmax'].isel(lat=lat_idx, lon=lon_idx).values
                    yearly_data_tasmax.append(tasmax)  # Append the data for the year
                    yearly_data_t.append(t)  # Append the time array for the year
                    print(f"✅ Successfully opened dataset: Model {model}, Scenario {scenario}, Year {year}, tasmax", flush = True)
                except Exception as e:
                    print(f"❌ Error opening dataset: Model {model}, Scenario {scenario}, Year {year}, tasmax. Error: {e}", flush = True)
                    continue

                # Open the dataset using the pyDAP client
                try:
                    s3_url = f"{base_url}{model}/{scenario}/r1i1p1f1/pr/pr_day_{model}_{scenario}_r1i1p1f1_gn_{year}.nc"
                    fs = fsspec.filesystem("https")
                    ds = xr.open_dataset(s3_url, engine="h5netcdf")
                    lat = ds['lat'].values  # Extract latitudes as a numpy array
                    lon = ds['lon'].values  # Extract longitudes as a numpy array
                    t = ds['time'].values  # Extract time as a numpy array
                    lat_idx = 10   # example index in the latitude array
                    lon_idx = 25   # example index in the longitude array
                    pr = ds['pr'].isel(lat=lat_idx, lon=lon_idx).values
                    yearly_data_tasmax.append(pr)  # Append the data for the year
                    print(f"✅ Successfully opened dataset: Model {model}, Scenario {scenario}, Year {year}, pr", flush = True)
                except Exception as e:
                    print(f"❌ Error opening dataset: Model {model}, Scenario {scenario}, Year {year}, pr. Error: {e}", flush = True)
                    continue                           
                                
            #creating the lat, lon, time array
            tasmax_3d = np.concatenate(yearly_data_tasmax, axis=0)
            pr_3d = np.concatenate(yearly_data_pr, axis=0)
            flat_time = np.concatenate(yearly_data_t)  # Combine all arrays into one
            eva_3d = thornthwaite(lat, tasmax_3d, flat_time)
            pe_3d = (pr_3d - eva_3d).dropna()  # calculate precipitation excess
            spei3 = si.spei(pe_3d, timescale=90, fit_freq="ME")
            spei12 = si.spei(pe_3d, timescale=365, fit_freq="ME")

            # Fit GPD and compute quantiles 
            results_3=fit_gpd_clustered_3d(
                data=spei3,
                quantile=0.95,
                return_periods_computed=TR,
                time_scale=1
            )
            
            results_12=fit_gpd_clustered_3d(
                data=spei12,
                quantile=0.95,
                return_periods_computed=TR,
                time_scale=1
            )
            print(f"Results for Model {model}, Scenario {scenario}, Time Frame {tf} computed successfully.", flush = True)

            scenario_idx = scenarios.index(scenario)
            model_idx = models.index(model)

            for (i, j), res in results_3.items():
                SPEI_3_quantiles[i, j, scenario_idx, ttf[tf_idx], model_idx,:] = res["Quantiles"]
                            
            for (i, j), res in results_12.items():
                SPEI_12_quantiles[i, j, scenario_idx, ttf[tf_idx], model_idx,:] = res["Quantiles"]
                
            # Define folder structure: path_save/Scenario/Model/Time_Frame/
            scenario_dir = os.path.join(path_save, scenario)
            model_dir = os.path.join(scenario_dir, model)
            time_frame_dir = os.path.join(model_dir, str(tf))
            os.makedirs(time_frame_dir, exist_ok=True)  # Ensure directories exist

            # Define the NetCDF file name and save path
            filename = f"SPEI_s{scenario}_m{model}_tf{tf}.nc"
            save_path = os.path.join(time_frame_dir, filename)

            # Create an xarray Dataset for saving the SPEI data
            ds = xr.Dataset(
                {
                    "SPEI_3_quantiles": (["i", "j", "return_period"], SPEI_3_quantiles[:, :, scenarios.index(scenario), tf_idx, :, models.index(model)]),
                    "SPEI_12_quantiles": (["i", "j", "return_period"], SPEI_12_quantiles[:, :, scenarios.index(scenario), tf_idx, :, models.index(model)]),
                },
                coords={
                    "i": range(SPEI_3_quantiles.shape[0]),  # Latitude index
                    "j": range(SPEI_3_quantiles.shape[1]),  # Longitude index
                    "return_period": TR,        # Return periods
                },
                attrs={
                    "description": "Standardized Precipitation Evapotranspiration Index (SPEI)",
                    "scenario": scenario,
                    "model": model,
                    "time_frame": tf,
                },
            )

            # Save the Dataset to a NetCDF file
            ds.to_netcdf(save_path)
            print(f"Saved SPEI data to: {save_path}", flush = True)
            sys.exit()

# End the timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds", flush = True)