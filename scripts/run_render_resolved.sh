#!/bin/bash

# Path to the render script
RENDER_SCRIPT="/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py"

# Folders to process
FOLDers=(
    "/Users/lehoangan/Downloads/merge_b/drive-download-20260731T041851Z-1-001/merged"
    "/Users/lehoangan/Downloads/merge_L/merged"
    "/Users/lehoangan/Downloads/merge_d/merged"
)

for FOLDER in "${FOLDers[@]}"; do
    echo "======================================"
    echo "Rendering $FOLDER"
    echo "======================================"
    python "$RENDER_SCRIPT" --output_dir "$FOLDER"
done
