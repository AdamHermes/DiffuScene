import json
import sys
from pathlib import Path

# Adapted KEYS to match the files in merge_b, merge_L, merge_d
KEYS = ["scene_ids", "class_labels", "translations", "sizes", "angles"]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def merge_physcene_jsons(paths, dedupe_by_scene_id=True):
    merged = {k: [] for k in KEYS}
    seen_scene_ids = set()
    n_skipped_dupes = 0

    for path in paths:
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping...")
            continue
            
        data = load_json(path)

        # sanity check: same keys, same number of scenes across all keys in this file
        n_scenes = len(data["scene_ids"])
        for k in KEYS:
            if k not in data:
                raise ValueError(f"{path} is missing key '{k}'")
            if len(data[k]) != n_scenes:
                raise ValueError(
                    f"{path}: key '{k}' has {len(data[k])} entries, "
                    f"but scene_ids has {n_scenes}. File is inconsistent."
                )

        print(f"Loaded {path}: {n_scenes} scenes")

        for i in range(n_scenes):
            scan_id = data["scene_ids"][i]

            if dedupe_by_scene_id and scan_id in seen_scene_ids:
                n_skipped_dupes += 1
                continue

            seen_scene_ids.add(scan_id)
            for k in KEYS:
                merged[k].append(data[k][i])

    print(f"\nMerged total: {len(merged['scene_ids'])} scenes")
    if n_skipped_dupes:
        print(f"Skipped {n_skipped_dupes} duplicate scene_id(s) "
              f"(same scene present in multiple input files)")

    return merged

def main():
    out_path = "/Users/lehoangan/Downloads/final_merged_collision_params.json"
    in_paths = [
        "/Users/lehoangan/Downloads/merge_b/drive-download-20260731T041851Z-1-001/merged/collision_params.json",
        "/Users/lehoangan/Downloads/merge_L/merged/collision_params.json",
        "/Users/lehoangan/Downloads/merge_d/merged/collision_params.json"
    ]

    merged = merge_physcene_jsons(in_paths)

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nWrote merged file to: {out_path}")

if __name__ == "__main__":
    main()
