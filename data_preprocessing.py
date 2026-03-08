import os
import cv2
import random
import shutil

# Number of images to select
NUM_IMAGES = 499

# Original dataset paths
parasitized_path = r'cell_images\Parasitized'
uninfected_path = r'cell_images\Uninfected'

# New subset folder
subset_base_path = r'\subset_dataset'
subset_parasitized = os.path.join(subset_base_path, 'Parasitized')
subset_uninfected = os.path.join(subset_base_path, 'Uninfected')

# Create new folders if they don’t exist
os.makedirs(subset_parasitized, exist_ok=True)
os.makedirs(subset_uninfected, exist_ok=True)

# ----------------------------
# Select 499 Parasitized Images
# ----------------------------
parasitized_images = [f for f in os.listdir(parasitized_path) if f.endswith(".png")]
selected_parasitized = random.sample(parasitized_images, NUM_IMAGES)

for img_name in selected_parasitized:
    src_path = os.path.join(parasitized_path, img_name)
    dst_path = os.path.join(subset_parasitized, img_name)
    shutil.copy(src_path, dst_path)

print("Parasitized images copied:", len(selected_parasitized))

# ----------------------------
# Select 499 Uninfected Images
# ----------------------------
uninfected_images = [f for f in os.listdir(uninfected_path) if f.endswith(".png")]
selected_uninfected = random.sample(uninfected_images, NUM_IMAGES)

for img_name in selected_uninfected:
    src_path = os.path.join(uninfected_path, img_name)
    dst_path = os.path.join(subset_uninfected, img_name)
    shutil.copy(src_path, dst_path)

print("Uninfected images copied:", len(selected_uninfected))

print("✅ Subset dataset created successfully!")