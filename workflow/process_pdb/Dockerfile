# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir biopython

# Copy the script to the container
COPY filter_fasta.py /app/

# Fix the __name__ line in the script
RUN sed -i 's/\*\*name\*\*/__name__/' /app/filter_fasta.py

# Entry point to run the script
ENTRYPOINT ["python", "/app/filter_fasta.py"]