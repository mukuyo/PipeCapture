import os
from PIL import Image

folder_path = "../PipeIsoGen/data/real/images/test/rgb/"

output_dir = "../PipeIsoGen/data/real/images/test/rgb/"

for filename in os.listdir(folder_path):
    if filename.endswith("jpg") or filename.endswith("jpeg"):
        img = Image.open(os.path.join(folder_path, filename))

        base_filename = os.path.splitext(filename)[0]
        
        img.save(os.path.join(output_dir, base_filename + ".png"))

        print(f"{filename} complete change")
