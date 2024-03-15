import os
import shutil

# Paths to the original train, val, and test directories
train_dir = "/nfs/hpc/share/joshishu/DL_Project/data/train"
val_dir = "/nfs/hpc/share/joshishu/DL_Project/data/valid"
test_dir = "/nfs/hpc/share/joshishu/DL_Project/data/test"

# Path to the new directory where you want to combine all the images
combined_dir = "/nfs/hpc/share/joshishu/DL_Project/combined_data"

# Create the combined directory if it doesn't exist
os.makedirs(combined_dir, exist_ok=True)

# Function to copy images from a source directory to the combined directory
def copy_images(src_dir, dst_dir):
    for class_name in os.listdir(src_dir):
        class_src_dir = os.path.join(src_dir, class_name)
        class_dst_dir = os.path.join(dst_dir, class_name)
        os.makedirs(class_dst_dir, exist_ok=True)
        for image_name in os.listdir(class_src_dir):
            src_image_path = os.path.join(class_src_dir, image_name)
            dst_image_path = os.path.join(class_dst_dir, image_name)
            shutil.copy(src_image_path, dst_image_path)

# Copy images from train, val, and test directories to the combined directory
copy_images(train_dir, combined_dir)
copy_images(val_dir, combined_dir)
copy_images(test_dir, combined_dir)

print("Images combined successfully!")
