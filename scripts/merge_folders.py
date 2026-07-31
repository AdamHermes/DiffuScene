import os
import json
import shutil
import sys
from pathlib import Path

def merge_folders(base_dir):
    base_path = Path(base_dir)
    merged_dir = base_path / 'merged'
    merged_dir.mkdir(exist_ok=True)
    
    merged_json = {
        "scene_ids": [],
        "class_labels": [],
        "translations": [],
        "sizes": [],
        "angles": []
    }
    
    # Iterate through run folders
    for run_folder in base_path.iterdir():
        if not run_folder.is_dir() or run_folder.name == 'merged':
            continue
            
        print(f"Processing {run_folder.name}...")
        
        # Merge JSON
        json_file = run_folder / 'collision_params.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                merged_json["scene_ids"].extend(data.get("scene_ids", []))
                merged_json["class_labels"].extend(data.get("class_labels", []))
                merged_json["translations"].extend(data.get("translations", []))
                merged_json["sizes"].extend(data.get("sizes", []))
                merged_json["angles"].extend(data.get("angles", []))
        
        # Merge other files and directories
        for item in run_folder.iterdir():
            if item.name == 'collision_params.json':
                continue
                
            if item.is_file():
                # Top level files (like .png)
                shutil.copy2(item, merged_dir / item.name)
            elif item.is_dir():
                # Subdirectories
                merged_sub_dir = merged_dir / item.name
                merged_sub_dir.mkdir(exist_ok=True)
                for sub_item in item.iterdir():
                    if sub_item.is_file():
                        shutil.copy2(sub_item, merged_sub_dir / sub_item.name)
                    elif sub_item.is_dir():
                        shutil.copytree(sub_item, merged_sub_dir / sub_item.name, dirs_exist_ok=True)

    # Write merged json
    with open(merged_dir / 'collision_params.json', 'w') as f:
        json.dump(merged_json, f, indent=2)

    print("Merging complete. Check 'merged' directory.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        merge_folders(sys.argv[1])
    else:
        merge_folders('.')
