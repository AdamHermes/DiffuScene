"""
Script to merge collision_params_*.json files in each room folder
(bedroom, diningroom, livingroom) into a single collision_params_merged.json.

Each collision_params file is a JSON dict with keys:
    scene_ids, class_labels, translations, sizes, angles
Each value is a list. Merging = concatenating lists across files.

Files are sorted by their numeric index to ensure deterministic ordering.
"""

import json
import glob
import os
import re


def extract_index(filepath):
    """Extract numeric index from filename like collision_params_30.json -> 30."""
    basename = os.path.basename(filepath)
    match = re.search(r'collision_params_(\d+)\.json', basename)
    if match:
        return int(match.group(1))
    return -1  # fallback for files without index


def merge_collision_params(room_dir):
    """Merge all collision_params_*.json files in a room directory."""
    pattern = os.path.join(room_dir, "collision_params_*.json")
    files = sorted(glob.glob(pattern), key=extract_index)
    # Exclude the output file itself to avoid reading previous merge results
    files = [f for f in files if "collision_params_merged" not in os.path.basename(f)]

    if not files:
        print(f"  [SKIP] No collision_params_*.json files found in {room_dir}")
        return

    print(f"  Found {len(files)} files to merge:")
    for f in files:
        print(f"    - {os.path.basename(f)}")

    # Initialize merged dict with empty lists
    merged = {
        "scene_ids": [],
        "class_labels": [],
        "translations": [],
        "sizes": [],
        "angles": [],
    }

    total_entries = 0
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        n_entries = len(data["scene_ids"])
        total_entries += n_entries
        print(f"    {os.path.basename(filepath)}: {n_entries} entries")

        for key in merged:
            if key not in data:
                print(f"    [WARNING] Key '{key}' not found in {filepath}")
                continue
            merged[key].extend(data[key])

    print(f"  Total merged entries (before dedup): {total_entries}")

    # Deduplicate: for duplicate scene_ids, keep the LAST entry.
    # Duplicates occur when generate_diffusion.py is re-run with the same
    # start_index without clearing the old JSON — the resume mechanism
    # loads existing data and appends new results on top.
    seen = {}
    for idx, sid in enumerate(merged["scene_ids"]):
        seen[sid] = idx  # later index overwrites earlier -> keeps last

    unique_indices = sorted(seen.values())
    n_before = len(merged["scene_ids"])
    n_after = len(unique_indices)
    n_removed = n_before - n_after

    if n_removed > 0:
        print(f"  Dedup: removed {n_removed} duplicate entries, {n_after} unique remain")
        for key in merged:
            merged[key] = [merged[key][i] for i in unique_indices]
    else:
        print(f"  No duplicates found")

    # Verify all lists have the same length
    lengths = {k: len(v) for k, v in merged.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        print(f"  [WARNING] Merged lists have different lengths: {lengths}")
    else:
        print(f"  Total unique entries: {n_after}")

    # Save merged file
    output_path = os.path.join(room_dir, "collision_params_merged.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved to: {output_path} ({file_size_kb:.1f} KB)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rooms = ["bedroom", "diningroom", "livingroom"]

    for room in rooms:
        room_dir = os.path.join(script_dir, room)
        print(f"\n{'='*60}")
        print(f"Processing: {room}")
        print(f"{'='*60}")

        if not os.path.isdir(room_dir):
            print(f"  [SKIP] Directory not found: {room_dir}")
            continue

        merge_collision_params(room_dir)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
