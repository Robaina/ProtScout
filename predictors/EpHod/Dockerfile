# Use NVIDIA CUDA base image compatible with PyTorch 1.7.0
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda-11.7/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH}"

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the repository contents to working directory
COPY ephod /app/ephod

# Set torch cache dir for esm weitghts
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    chmod -R 777 /root/.cache

# Set up entrypoint
ENTRYPOINT ["python3", "/app/ephod/run.py"]