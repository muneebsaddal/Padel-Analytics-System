import os
import json

# Root path where your videos/clips start
root_path = '/workspace/Padel-Analytics-System/data/youtube_batch/'
output_file = 'manifest.jsonl'
valid_extensions = ('.jpg', '.jpeg', '.png')

with open(output_file, 'w') as f:
    # CVAT Header
    f.write(json.dumps({"type": "images", "version": "1.1"}) + '\n')
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(root_path):
        for file in sorted(files):
            if file.lower().endswith(valid_extensions):
                # Calculate the relative path from the root_path
                relative_path = os.path.relpath(os.path.join(root, file), root_path)
                f.write(json.dumps({"name": relative_path}) + '\n')

print(f"Done! Created {output_file}")

