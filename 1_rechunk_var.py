import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format*",
    category=UserWarning,
)


def reencode_file(nc_path, model, scenario, zarr_path):
    ds = xr.open_dataset(nc_path).compute()
    # Convert to standard calendar in datetime64[ns] and interpolate
    # e.g. from cftime.DatetimeNoLeap in NorESM2-MM
    # or cftime.Datetime360Day in HadGEM3-GC31-LL
    ds = ds.convert_calendar("standard", align_on="year", dim="time", use_cftime=False)
    full_coords = np.arange(
        ds.time.values[0],
        ds.time.values[-1] + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]") + np.timedelta64(12, "h")
    ds = ds.interp(time=full_coords, method="linear")

    ds = ds.chunk({"time": -1, "lat": 120, "lon": 120}).compute()
    ds = ds.expand_dims(
        model=[model],
        scenario=[scenario],
    )
    ds.to_zarr(zarr_path, region="auto")


def read_meta(meta_fname, filter_var):
    meta = pd.read_csv(meta_fname).query(f"variable == '{filter_var}'")
    meta.year = meta.year.astype(int)
    meta = meta[meta.year.isin(range(1984, 2091))].sort_values(by="year")
    return meta


if __name__ == "__main__":
    logging.basicConfig(
        **{
            "level": "INFO",
            "format": "%(asctime)s %(levelname)-8s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    )
    work_path = Path(sys.argv[1])
    nc_path = Path(sys.argv[2])
    var = sys.argv[3]
    job = int(sys.argv[4])
    logging.info(
        "Start with work_path=%s nc_path=%s, var=%s job=%d",
        work_path,
        nc_path,
        var,
        job,
    )

    meta = read_meta(work_path / "file_metadata.csv", var)
    zarr_path = work_path / "nex-gddp-cmip6.zarr"

    for i, ((model, scenario), df) in enumerate(meta.groupby(["model", "scenario"])):
        if i != job:
            continue
        logging.info("%d Processing %s %s with %d files", i, model, scenario, len(df))

        fnames = df.path

        for fname in fnames:
            logging.info("Processing %s", fname)
            reencode_file(nc_path / fname, model, scenario, zarr_path)

        logging.info(f"Done for {model=}, {scenario=}")
