# ------------------------------------------------------------------------------
# 1) Use the PyTorch Docker image with Python 3.8
# ------------------------------------------------------------------------------
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

# Avoid interactive dialogs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# ------------------------------------------------------------------------------
# 2) Install any system packages you need
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libreadline-dev \
    libsqlite3-dev \
    libbz2-dev \
    curl \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 3) Install Rust (avoid using 'source' which /bin/sh doesn't recognize)
# ------------------------------------------------------------------------------
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default stable

# ------------------------------------------------------------------------------
# 4) Upgrade pip, setuptools, wheel
# ------------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ------------------------------------------------------------------------------
# 5) Install PyTorch Geometric libs for PyTorch 1.12.0 + cu113
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    pyg-lib \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# ------------------------------------------------------------------------------
# 6) Install your other Python deps
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    biopython \
    sentencepiece \
    transformers \
    modelcif \
    "fair-esm[esmfold]" \
    'dllogger @ git+https://github.com/NVIDIA/dllogger.git'

# ------------------------------------------------------------------------------
# 7) Copy your code, install OpenFold, etc.
# ------------------------------------------------------------------------------
COPY GeoPoc /app/GeoPoc

WORKDIR /app/GeoPoc/openfold
RUN python setup.py install

WORKDIR /app/GeoPoc
RUN chmod +x feature_extraction/mkdssp

# ------------------------------------------------------------------------------
# 8) Environment variables & entrypoint
# ------------------------------------------------------------------------------
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "predict.py"]
CMD ["-h"]
