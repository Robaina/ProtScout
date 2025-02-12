FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3-pip \
    python3-dev \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app/catpred

# Copy application files and set permissions
COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
# Build and install wheel
RUN pip install wheel && \
    python3 setup.py bdist_wheel && \
    pip install dist/*.whl

# Set proper permissions
RUN chmod -R 755 /app/catpred && \
    chown -R root:root /app/catpred

# Download and extract pre-trained models and databases
RUN wget https://catpred.s3.amazonaws.com/production_models.tar.gz -q \
    && wget https://catpred.s3.amazonaws.com/processed_databases.tar.gz -q \
    && tar -xzf production_models.tar.gz \
    && tar -xzf processed_databases.tar.gz \
    && rm production_models.tar.gz processed_databases.tar.gz
    
# Set permissions for models and databases
RUN chmod -R 755 /app/catpred/production_models && \
    chmod -R 755 /app/catpred/processed_databases
    
# Set torch cache dir for esm weitghts
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    chmod -R 777 /root/.cache

# Create input/output directories with proper permissions
RUN mkdir -p /input /output && \
    chmod -R 777 /input && \
    chmod -R 777 /output

ENTRYPOINT ["python3", "predict_kinetics.py"]