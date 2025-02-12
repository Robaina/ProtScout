# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    make \
    gcc \
    g++ \
    pkg-config \
    freetype* \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the local repository contents
COPY . /app/

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Set torch cache dir for esm weitghts
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    chmod -R 777 /root/.cache

# Default command to run predictions
ENTRYPOINT ["python", "entrypoint.py"]