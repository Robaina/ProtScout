# Build the Docker image
docker build -t prepare_inputs .

# Run the container with mounted volumes for data access
docker run -v /path/to/fasta_dir:/data/fasta \
           -v /path/to/substrate.tsv:/data/substrate.tsv \
           -v /path/to/output:/data/output \
           prepare_inputs \
           --fasta_dir /data/fasta \
           --substrate_tsv /data/substrate.tsv \
           --output_dir /data/output