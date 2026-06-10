#!/bin/bash

set -e

URL="https://www.ipb.uni-bonn.de/html/projects/mai_city/mai_city.tar.gz"
OUT_DIR="../maicity"
FILE_NAME="mai_city.tar.gz"

mkdir -p "$OUT_DIR"

echo "Downloading MaiCity dataset..."
wget -c "$URL" -O "$OUT_DIR/$FILE_NAME"

echo "Extracting..."
tar -xzf "$OUT_DIR/$FILE_NAME" -C "$OUT_DIR"
rm "$OUT_DIR/$FILE_NAME"

echo "Done. Dataset is saved in: $OUT_DIR"