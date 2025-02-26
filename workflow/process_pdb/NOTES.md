# Save your script as filter_fasta.py first

# Build the Docker image
docker build -t filter-fasta .

# Run the container with mounted volumes for data access
docker run -v /path/to/faa_dir:/data/faa \
           -v /path/to/pdb_dir:/data/pdb \
           filter-fasta \
           --faa_dir /data/faa \
           --pdb_dir /data/pdb