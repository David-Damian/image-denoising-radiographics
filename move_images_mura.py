import os
import shutil

train_origin_path = os.path.join("MURA-v1.1", "train_image_paths.csv")
valid_origin_path = os.path.join("MURA-v1.1", "valid_image_paths.csv")

print(train_origin_path)

train_dest_path = "DL_images/train"
valid_dest_path = "DL_images/valid"

with open(train_origin_path) as f:
    img_num = 1
    lines = f.readlines()
    for line in lines:
        origin = line.strip()
        if os.path.exists(origin):
            shutil.move(origin, f'{train_dest_path}/image_{img_num}.png')
            img_num += 1
f.close()

with open(valid_origin_path) as g:
    img_num = 1
    lines = g.readlines()
    for line in lines:
        origin = line.strip()
        if os.path.exists(origin):
            shutil.move(origin, f'{valid_dest_path}/image_{img_num}.png')
            img_num += 1
g.close()