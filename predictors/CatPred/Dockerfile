# Dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive dialog during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3-pip \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Optional: Install mamba for faster dependency resolution
RUN conda install -c conda-forge mamba -y

# Clone CatPred repository
WORKDIR /app
RUN git clone https://github.com/maranasgroup/catpred.git
WORKDIR /app/catpred

# Create conda environment and install dependencies
RUN mamba env create -f environment.yml || conda env create -f environment.yml \
    && conda init bash \
    && echo "conda activate catpred" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# Install additional Python packages
RUN pip install -e . && \
    pip install ipdb fair-esm rotary_embedding_torch==0.6.5 egnn_pytorch -q

# Download and extract pre-trained models and databases
RUN wget https://catpred.s3.amazonaws.com/production_models.tar.gz -q \
    && wget https://catpred.s3.amazonaws.com/processed_databases.tar.gz -q \
    && tar -xzf production_models.tar.gz \
    && tar -xzf processed_databases.tar.gz \
    && rm production_models.tar.gz processed_databases.tar.gz

# Create input/output directories
RUN mkdir -p /input /output /results

# Copy prediction script
COPY catpred /app/catpred
COPY demo_run.py /app/catpred/demo_run.py
COPY entrypoint.sh /app/catpred/entrypoint.sh
RUN chmod +x /app/catpred/entrypoint.sh

ENTRYPOINT ["/app/catpred/entrypoint.sh"]