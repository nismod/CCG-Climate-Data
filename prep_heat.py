from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


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

    tasks = pd.concat(all_task_dfs).reset_index(drop=True)

    epoch_scenarios = [
        pd.DataFrame({"epoch": ["1984-2014"], "scenario": ["historical"]})
    ]
    for scenario in ("ssp126", "ssp245", "ssp370", "ssp585"):
        epoch_scenarios.append(
            pd.DataFrame(
                {
                    "epoch": ["2015-2040", "2041-2065", "2066-2090"],
                    "scenario": [scenario, scenario, scenario],
                }
            )
        )
    epoch_scenario = pd.concat(epoch_scenarios)
    return tasks.merge(epoch_scenario, on="scenario", how="left")


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


def setup_store(fname, models, scenarios, rps, epochs):
    ds = xr.open_dataset(fname).drop_dims("time")
    ds = ds.expand_dims(
        model=models, scenario=scenarios, return_period=rps, epoch=epochs
    )
    da = xr.DataArray(
        coords={
            "model": ds.coords["model"],
            "scenario": ds.coords["scenario"],
            "return_period": ds.coords["return_period"],
            "epoch": ds.coords["epoch"],
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
        },
        dims=["model", "scenario", "return_period", "epoch", "lat", "lon"],
    )
    ds["tasmax_return_level"] = da
    # ds.to_zarr(f"nex-gddp-cmip6.{model}.{scenario}.zarr", compute=False)
    return ds


lat_cells = 600  # Number of latitude cells
lon_cells = 1440  # Number of longitude cells
grid_size = 10  # Number of divisions along each axis

tasks = read_meta(Path(), grid_size**2)
scenarios = sorted(tasks.scenario.unique())
models = sorted(tasks.model.unique())
rps = [20, 50, 100, 200, 500, 1500]
tfs = ["1984-2014", "2015-2040", "2041-2065", "2066-2090"]
template_fname = (
    Path("./data") / "tasmax_day_ACCESS-CM2_historical_r1i1p1f1_gn_1984_v2.0.nc"
)
ds = setup_store(template_fname, models, scenarios, rps, tfs)

ds.to_zarr(f"tasmax_quantiles.zarr", mode="w", compute=False)


tasks = add_bounds(tasks, grid_size)
tasks.to_csv("tasmax_tasks.csv")
