import warnings
import pydap.client
import requests
import sys
import numpy as np
from scipy.stats import genpareto
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
import dask
from dask.array import map_blocks
import dask.array as da
import spei as si  # si for standardized index

path_save = "/soge-home/users/cenv0972/CCG/Heat/CMIP6/"  # Path to save the results

warnings.filterwarnings("ignore", message="PyDAP was unable to determine the DAP protocol")

# Setup S3 file system
fs = s3fs.S3FileSystem(anon=True)

# URLs of the text files
urls = ["https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v2.0_md5.txt"]

# Lists to store extracted information
tasmax_files = []  
tasmin_files = []  
pr_files = [] 
models = set()  
scenarios = set()  
variables = set() 
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
base_url = "s3://nex-gddp-cmip6/"

# Define the return periods to calculate
TR = [20, 50, 100, 200, 500, 1500]

# Initialize arrays with NaN values
SPEI_3_quantiles = np.full((lat_cells, lon_cells, len(scenarios), num_time_frames, len(models), num_TR_computed), np.nan)
SPEI_12_quantiles = np.full((lat_cells, lon_cells, len(scenarios), num_time_frames, len(models), num_TR_computed), np.nan)

try:
    job_num = int(sys.argv[1])
    total_num = int(sys.argv[2])
except IndexError:
    print(f"Usage: python {__file__} <start_num> <end_num>")
    sys.exit()

grid_div=25  # Number of grid divisions
vector_indices = np.arange(job_num,len(models)*len(scenarios)*grid_div,total_num)
indices_grid = np.floor(vector_indices/(len(models)*len(scenarios))).astype(int)
indices_models = np.floor((vector_indices-indices_grid*(len(models)*len(scenarios)))/(len(scenarios))).astype(int)
indices_scenarios = np.floor((vector_indices-indices_grid*(len(models)*len(scenarios))-indices_models*len(scenarios))).astype(int)

grid_size = int(np.sqrt(grid_div)) 

# Define the latitude and longitude ranges
lat_range = (-60, 90)  
lon_range = (0, 360)  

# Divide the latitude and longitude into equal intervals
lat_intervals = np.linspace(lat_range[0], lat_range[1], grid_size + 1)
lon_intervals = np.linspace(lon_range[0], lon_range[1], grid_size + 1)

indices_models = indices_models.astype(int)
indices_scenarios = indices_scenarios.astype(int)
indices_models = indices_models.tolist()
indices_scenarios = indices_scenarios.tolist()
models = [models[i] for i in indices_models]
scenarios= [scenarios[i] for i in indices_scenarios]
print('Starting the script to process climate data...', flush = True)


def compute_return_levels_1d(series, quantile, return_periods, time_scale):
    series = np.asarray(series)
    valid_series = series[~np.isnan(series)]
    threshold = np.quantile(valid_series, quantile)
    exceedances = valid_series[valid_series > threshold]
    
    if len(exceedances) < 5:
        print(f"Not enough exceedances for quantile {quantile} in series {series}. Returning NaN.", flush = True)
        return np.full(len(return_periods), np.nan)
        
    excesses = exceedances - threshold
    shape, loc, scale = genpareto.fit(excesses)
    
    quantiles = [
        threshold + genpareto.ppf(1 - 1 / (t * time_scale), shape, loc, scale)
        for t in return_periods
    ]
    
    return np.array(quantiles)


def thornthwaite(lat, T, time):
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

def filter_files(files, model, scenario, ensemble_member, variable, year_strings):
    return [
        file for file in files
        if model in file and scenario in file and ensemble_member in file and variable in file
        and any(year in file for year in year_strings)
    ]

# Wrapper to compute return levels per block while preserving spatial dims
def _return_levels_block(data1, data2, time, lat, quantile, tr, timescale):
    if data1.size == 0:  # Check if the block is empty
        print("Warning: Empty block encountered. Skipping.", flush=True)
        return np.full((len(tr),) + data1.shape[1:], np.nan)
    
    data1 = np.asarray(data1)  # Convert to NumPy array
    data2 = np.asarray(data2)  # Convert to NumPy array
    t, ny, nx = data1.shape
    out = np.full((len(tr), ny, nx), np.nan, dtype=np.float64)
    
    # Calculate Thornthwaite evapotranspiration
    eva = thornthwaite(lat, data1, time)
    
    # Calculate precipitation excess
    pe = data2 - eva
    pe[np.isnan(pe)] = 0  # Replace NaN values with 0
    
    # Calculate SPEI
    spei = si.spei(pe, timescale, fit_freq="ME")
    
    # Compute return levels for each grid point
    for j in range(ny):
        for i in range(nx):
            ts = spei[:, j, i]
            if np.all(np.isnan(ts)):
                continue
            out[:, j, i] = compute_return_levels_1d(ts, quantile, tr, timescale)
    
    return out

# Loop through all models and scenarios
for grid_idx in indices_grid:
    lat_idx = grid_idx // grid_size  
    lon_idx = grid_idx % grid_size   
    lat_idx = lat_idx.astype(int)  
    lon_idx = lon_idx.astype(int)  
    lat_bounds = (lat_intervals[lat_idx], lat_intervals[lat_idx + 1])
    lon_bounds = (lon_intervals[lon_idx], lon_intervals[lon_idx + 1])
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
                matching_files = [
                    file for file in tasmax_files
                    if model in file and scenario in file and file.endswith("_v2.0.nc")
                ]
                print(f"Reading year {tf} for Model {model}, Scenario {scenario}", flush = True)
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
                ds = xr.open_mfdataset(filehandles, preprocess=preprocessor, parallel=True, engine="h5netcdf",chunks={"time": 100})
                tasmax = ds['tasmax']
                matching_files = [
                    file for file in pr_files
                    if model in file and scenario in file and file.endswith("_v2.0.nc")
                ]
                print(f"Reading year {tf} for Model {model}, Scenario {scenario}", flush = True)
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
                ds = xr.open_mfdataset(filehandles, preprocess=preprocessor, parallel=True, engine="h5netcdf",chunks={"time": 100})
                pr = ds['pr']
                tt = ds['time'].values

                # print('Sample of tasmax data:', tasmax.isel(time=1000, lat=2, lon=2).values, flush = True)
                lat_coords = ds['lat'].values
                lon_coords = ds['lon'].values
                if tasmax.size == 0 or np.isnan(tasmax).all():
                    print("Warning: All values in tasmax are NaN. Skipping computation.", flush=True)
                    final_result = np.full((len(TR), len(lat_coords), len(lon_coords)), np.nan)
                else:
                    # Ensure proper chunking
                    tasmax = tasmax.chunk({'time': -1, 'lat': 1, 'lon': 1})
                    pr = pr.chunk({'time': -1, 'lat': 1, 'lon': 1})

                    # Use map_blocks to apply the function to Dask arrays
                    result3 = da.map_blocks(
                        _return_levels_block,       
                        tasmax.data,                
                        pr.data,                    
                        tt, lat_coords, 0.95, TR, 90,  
                        dtype="f8",                 
                        chunks=( (len(TR),), ) + tuple(tasmax.data.chunks[1:])  
                    )
                    result12 = da.map_blocks(
                        _return_levels_block,       
                        tasmax.data,                
                        pr.data,                    
                        tt, lat_coords, 0.95, TR, 360, 
                        dtype="f8",                 
                        chunks=( (len(TR),), ) + tuple(tasmax.data.chunks[1:])  
                    )

                    final_result3 = result3.compute()
                    final_result12 = result12.compute()

                # No manual reshape; final_result is expected (len(TR), lat, lon)
                result_xr3 = xr.DataArray(
                    final_result3,
                    dims=["return_period", "lat", "lon"],
                    coords={
                        "return_period": TR,
                        "lat": lat_coords,
                        "lon": lon_coords
                    },
                    name="spei3_return_level"
                )

                result_xr12 = xr.DataArray(
                    final_result12,
                    dims=["return_period", "lat", "lon"],
                    coords={
                        "return_period": TR,
                        "lat": lat_coords,
                        "lon": lon_coords
                    },
                    name="spei12_return_level"
                )

                # Define folder structure: path_save/Scenario/Model/Time_Frame/
                scenario_dir = os.path.join(path_save, scenario)
                model_dir = os.path.join(scenario_dir, model)
                time_frame_dir = os.path.join(model_dir, str(tf))
                os.makedirs(time_frame_dir, exist_ok=True) 
                output_folder = os.path.join(path_save, scenario, model, str(tf) + '/')
                os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
                # Calculate quantiles for each return period
                quantiles_to_plot3 = result_xr3.values[:, :, :]
                quantiles_to_plot12 = result_xr12.values[:, :, :]
                
                # Ensure the quantiles_to_plot is a 2D array for plotting
                result_xr3.to_netcdf(os.path.join(output_folder, f"spei3_quantiles_{model}_{scenario}_{tf}_n_{job_num}.nc"))
                result_xr12.to_netcdf(os.path.join(output_folder, f"spei12_quantiles_{model}_{scenario}_{tf}_n_{job_num}.nc"))
                print(f"Saved quantiles", flush = True)
