import os
import random
import shutil

# Path to the combined dataset directory
combined_dir = "/nfs/hpc/share/joshishu/DL_Project/BirdClass_Robustness/Adversarial_Wide/PGD_Images_n=21/Combined_adv_orig"

# Function to shuffle images within each class directory
def shuffle_images_in_directory(directory_path):
    for class_name in os.listdir(directory_path):
        class_dir = os.path.join(directory_path, class_name)
        
        image_filenames = os.listdir(class_dir)
        random.shuffle(image_filenames)
        
        temp_dir = os.path.join(directory_path, f"temp_{class_name}")
        os.makedirs(temp_dir, exist_ok=True)
        
        for idx, filename in enumerate(image_filenames):
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join(temp_dir, f"{idx:05d}_{filename}")
            shutil.copy(src_path, dst_path)
        
        shutil.rmtree(class_dir)
        os.rename(temp_dir, class_dir)

# Shuffle images in each class directory within the combined directory
shuffle_images_in_directory(combined_dir)

print("Images within each class directory have been shuffled successfully.")
