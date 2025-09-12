import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format*",
    category=UserWarning,
)


def reencode_file(nc_path, model, scenario, zarr_path):
    ds = xr.open_dataset(nc_path).compute()
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
        # if i != job:
        #     continue
        logging.info("%d Processing %s %s with %d files", i, model, scenario, len(df))

        # fnames = df.path

        # for fname in tqdm(fnames):
        #     reencode_file(nc_path / fname, model, scenario, zarr_path)

        # logging.info(f"Done for {model=}, {scenario=}")
