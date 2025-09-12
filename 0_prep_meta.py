import logging
import sys
import warnings
from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import requests
import xarray as xr
from zarr.errors import UnstableSpecificationWarning

warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The data type*",
    category=UnstableSpecificationWarning,
)


def download_file(url, fname):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def read_meta(nc_path, meta_fname):
    """Download (if necessary) and read NetCDF file metadata"""
    if meta_fname.exists():
        return pd.read_csv(meta_fname)

    # Read or download NEX-GDDP-CMIP6 index file
    nc_meta_url = "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v2.0_md5.txt"
    nc_meta_fname = nc_path / "index_v2.0_md5.txt"

    if not nc_meta_fname.exists():
        download_file(nc_meta_url, nc_meta_fname)

    meta = pd.read_csv(
        nc_meta_fname, sep=" ", header=None, names=["hash", "_", "path"]
    ).drop(columns="_")

    # Filter to ensure only v2.0 files
    meta = meta[meta.path.str.endswith("_v2.0.nc")]
    # Parse model, scenario, variable and year from filenames
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
    # Save as csv
    meta.to_csv(meta_fname)

    return meta


def meta_to_tasks(meta, vars, grid_size):
    meta = meta[meta.variable.isin(vars)].copy()
    model_scenario_combinations = (
        meta[["model", "scenario"]]
        .drop_duplicates()
        .sort_values(by=["model", "scenario"])
        .reset_index(drop=True)
    )
    all_task_dfs = []
    nsubgrids = grid_size**2
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
                    "scenario": [scenario] * 3,
                }
            )
        )
    epoch_scenario = pd.concat(epoch_scenarios)
    tasks = tasks.merge(epoch_scenario, on="scenario", how="left")
    return add_bounds(tasks, grid_size)


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


def setup_store(template_fname, models, scenarios):
    ds = xr.open_dataset(template_fname)

    time = np.arange(
        np.datetime64("1984-01-01"), np.datetime64("2091-01-01"), dtype="datetime64[D]"
    ).astype("datetime64[ns]") + np.timedelta64(12, "h")

    da = xr.DataArray(
        dask.array.zeros(
            (
                len(models),
                len(scenarios),
                len(ds.coords["lat"]),
                len(ds.coords["lon"]),
                len(time),
            ),
            chunks=(1, 1, 120, 120, 365),
        ),
        coords={
            "model": models,
            "scenario": scenarios,
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
            "time": time,
        },
        dims=["model", "scenario", "lat", "lon", "time"],
    )
    zero_ds = xr.Dataset({"pr": da, "tasmax": da})
    return zero_ds


def setup_rp_store(template_fname, models, scenarios, rps, epochs):
    ds = xr.open_dataset(template_fname).drop_dims("time")
    da = xr.DataArray(
        dask.array.zeros(
            (
                len(models),
                len(scenarios),
                len(rps),
                len(epochs),
                len(ds.coords["lat"]),
                len(ds.coords["lon"]),
            ),
            chunks=(1, 1, len(rps), 1, 120, 120),
        ),
        coords={
            "model": models,
            "scenario": scenarios,
            "return_period": rps,
            "epoch": epochs,
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
        },
        dims=["model", "scenario", "return_period", "epoch", "lat", "lon"],
    )
    ds["tasmax_return_level"] = da
    ds["spei3_return_level"] = da
    ds["spei12_return_level"] = da
    return ds


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    nc_path = Path(sys.argv[1])
    work_path = Path(sys.argv[2])
    logging.info(f"Start with {nc_path=}, {work_path=}")
    # sys.exit()

    grid_size = 10  # Number of divisions along each axis

    meta = read_meta(work_path, work_path / "file_metadata.csv")
    tasmax_tasks = meta_to_tasks(meta, ("tasmax",), grid_size)
    tasmax_tasks.to_csv(work_path / "tasmax_tasks.csv")

    spei_tasks = meta_to_tasks(meta, ("tasmax", "pr"), grid_size)
    spei_tasks.to_csv(work_path / "spei_tasks.csv")

    all_tasks = pd.concat([tasmax_tasks, spei_tasks])
    scenarios = sorted(all_tasks.scenario.unique())
    models = sorted(all_tasks.model.unique())
    rps = [20, 50, 100, 200, 500, 1500]
    epochs = ["1984-2014", "2015-2040", "2041-2065", "2066-2090"]

    template_fname = (
        nc_path
        / "NEX-GDDP-CMIP6"
        / "ACCESS-CM2"
        / "historical"
        / "r1i1p1f1"
        / "tasmax"
        / "tasmax_day_ACCESS-CM2_historical_r1i1p1f1_gn_1984_v2.0.nc"
    )

    store_path = work_path / "nex-gddp-cmip6.zarr"
    logging.info("Setting up data store at %s", store_path)
    ds = setup_store(template_fname, models, scenarios)
    ds.to_zarr(store_path, mode="w", compute=False)

    rp_store_path = work_path / "nex-gddp-cmip6.return_levels.zarr"
    logging.info("Setting up RP store at %s", rp_store_path)
    ds = setup_rp_store(template_fname, models, scenarios, rps, epochs)
    ds.to_zarr(rp_store_path, mode="w", compute=False)
    logging.info("Done")
