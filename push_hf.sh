#!/bin/bash
set -e
SPACE_DIR="../neuroscan_space"

mkdir -p "$SPACE_DIR/package"

# Shared package files
cp package/config.py         "$SPACE_DIR/package/config.py"
cp package/visualization.py  "$SPACE_DIR/package/visualization.py"
cp package/explainability.py "$SPACE_DIR/package/explainability.py"
cp index.html                "$SPACE_DIR/index.html"

# HF-specific files
cp deploy/hf/Dockerfile        "$SPACE_DIR/Dockerfile"
cp deploy/hf/main.py           "$SPACE_DIR/main.py"
cp deploy/hf/requirements.txt  "$SPACE_DIR/requirements.txt"

cd "$SPACE_DIR"
git add .
git commit -m "deploy: $(date '+%Y-%m-%d %H:%M')"
git push