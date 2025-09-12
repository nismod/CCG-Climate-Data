# Heat and drought hazard analysis

Download, preprocess NEX-GDDP-CMIP6 data, then calculate return periods for daily maximum
near-surface temperate (tasmax) and standardised precipitation evapotranspiration index (spei).

Usage:

```bash
# Download pr and tasmax variables ~4TB
# bash download_var.sh

# Set up metadata and empty ZARR stores
python 0_prep_meta.py /mnt/linux-filestore/data/incoming/NEX-GDDP-CMIP6/ /mnt/linux-filestore/mistral/ccg-2025-hazards

# Set up ZARR stores in working directory - run as array job for pr 0-163 and tasmax 0-145
python 1_rechunk_var.py /mnt/linux-filestore/mistral/ccg-2025-hazards/ /mnt/linux-filestore/data/incoming/NEX-GDDP-CMIP6/ tasmax 0
```
