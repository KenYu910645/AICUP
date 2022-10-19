import os
import shutil
import glob
import random
from shutil import rmtree

TRAIN_SPLIT = 0.8 # Propotion of the training dataset

CROP_NAME = ['banana', 'carrot', 'corn', 'dragonfruit', 'garlic',
             'guava', 'peanut', 'pineapple', 'pumpkin',
             'rice', 'soybean', 'sugarcane', 'tomato', 'bareland']

RAW_IMG_PATH = "/home/lab530/KenYu/aicpu/data/raw_img/"
TRAIN_DIR = "/home/lab530/KenYu/aicpu/data/training/"
VAL_DIR = "/home/lab530/KenYu/aicpu/data/validation/"

img_name_list = []
for crop in CROP_NAME:
    img_name_list += glob.glob(os.path.join(RAW_IMG_PATH, crop, "*.JPG"))
    img_name_list += glob.glob(os.path.join(RAW_IMG_PATH, crop, "*.jpg"))

print(f"Total image in raw dataset = {len(img_name_list)}")

random.shuffle(img_name_list)

train_imgs = img_name_list[:int(TRAIN_SPLIT*len(img_name_list))]
val_imgs   = img_name_list[ int(TRAIN_SPLIT*len(img_name_list)):]

print(f"Number of images in training set: {len(train_imgs)}")
print(f"Number of images in validation set: {len(val_imgs)}")

# Clean output directory 
if os.path.isdir(TRAIN_DIR):
    rmtree(TRAIN_DIR)
os.mkdir(TRAIN_DIR)
if os.path.isdir(VAL_DIR):
    rmtree(VAL_DIR)
os.mkdir(VAL_DIR)

for i, img_path in enumerate(train_imgs):
    cls = img_path.split('/')[-2]
    new_fn = os.path.join(TRAIN_DIR, f"{CROP_NAME.index(cls)}_{i}.jpg")
    shutil.copyfile(img_path, os.path.join(TRAIN_DIR, new_fn))
    print(f"({i}/{len(train_imgs)})")   
print(f"Copied {len(train_imgs)} image to {TRAIN_DIR}")

for i, img_path in enumerate(val_imgs):
    cls = img_path.split('/')[-2]
    new_fn = os.path.join(VAL_DIR, f"{CROP_NAME.index(cls)}_{i}.jpg")
    shutil.copyfile(img_path, os.path.join(VAL_DIR, new_fn))
    print(f"({i}/{len(val_imgs)})")
print(f"Copied {len(val_imgs)} image to {VAL_DIR}")



