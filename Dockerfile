# STRONG-RMMD — reproducible environment for evaluation and the paper figures.
#
#   docker build -t strong-rmmd .
#
# Reproduce the paper figures + tables (CPU is fine — reads the committed result JSONs):
#   docker run --rm -v "$PWD:/app" strong-rmmd \
#     jupyter nbconvert --to notebook --execute --inplace STRONG_RMMD/notebooks_paper/paper_figures.ipynb
#
# Training / ablations / theorems need a GPU + a CUDA-enabled torch (swap the base image for an
# nvidia/cuda one and reinstall torch from the CUDA index); the slim CPU image below is enough to
# rebuild every figure from the committed result JSONs.
FROM python:3.11-slim

WORKDIR /app

# netCDF4 needs the system HDF5/netCDF libraries; build-essential for any wheels that compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libhdf5-dev libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# `strong_rmmd` is imported as a top-level package (needs STRONG_RMMD/ on the path); the DGKNet
# baseline + data pipeline are imported as `dgknet_baseline...` (needs the repo root).
ENV PYTHONPATH=/app/STRONG_RMMD:/app

CMD ["bash"]
