#!/bin/bash
# Downloads two months of NYC TLC yellow taxi records (about 50 MB each).
set -e
mkdir -p data
for m in 2024-01 2024-02; do
  curl -fL -o data/yellow_tripdata_${m}.parquet \
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_${m}.parquet
done
echo "Done. Files are in data/"
