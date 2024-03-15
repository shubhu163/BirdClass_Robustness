import os
import random
import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchattacks
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
big_dataset_dir = "/nfs/hpc/share/joshishu/DL_Project/combined_data"
adversarial_dir = "/nfs/hpc/share/joshishu/DL_Project/Adv_Images"
combined_dir = "/nfs/hpc/share/joshishu/DL_Project/Combined_adv_orig"

# Load the big dataset
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
big_dataset = datasets.ImageFolder(big_dataset_dir, transform=transform)
big_dataloader = DataLoader(big_dataset, batch_size=1, shuffle=False)

weights = Wide_ResNet50_2_Weights.DEFAULT
model = wide_resnet50_2(weights=weights)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 525),  # Adjust to your number of species
    nn.LogSoftmax(dim=1)
)

model.load_state_dict(torch.load(f="/nfs/hpc/share/joshishu/DL_Project/pretrain_wide/fine_best_model_birds.pth"))
model = model.to(device)
model.eval()


# Define the attack
epsilon = 0.03  # Perturbation level
# attack = torchattacks.FGSM(model, eps=epsilon)
attack = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)


# Generate adversarial images from half of the big dataset
os.makedirs(adversarial_dir, exist_ok=True)
for i, (images, labels) in enumerate(big_dataloader):
    if i >= len(big_dataset) // 2:
        break
    images, labels = images.to(device), labels.to(device)
    adv_images = attack(images, labels)

    class_name = big_dataset.classes[labels.item()]
    os.makedirs(os.path.join(adversarial_dir, class_name), exist_ok=True)
    adv_image_path = os.path.join(adversarial_dir, class_name, f"adv_{i}.png")
    torchvision.utils.save_image(adv_images[0], adv_image_path)

# Combine adversarial images with the other half of original images
os.makedirs(combined_dir, exist_ok=True)
for class_name in os.listdir(big_dataset_dir):
    class_dir = os.path.join(big_dataset_dir, class_name)
    combined_class_dir = os.path.join(combined_dir, class_name)
    os.makedirs(combined_class_dir, exist_ok=True)
    image_names = os.listdir(class_dir)
    half = len(image_names) // 2
    for image_name in image_names[:half]:  # Copy original images
        shutil.copy(os.path.join(class_dir, image_name), combined_class_dir)
    if os.path.exists(os.path.join(adversarial_dir, class_name)):
        for image_name in os.listdir(os.path.join(adversarial_dir, class_name)):  # Copy adversarial images
            shutil.copy(os.path.join(adversarial_dir, class_name, image_name), combined_class_dir)

# Shuffle the combined dataset
# You can shuffle the dataset when loading it with DataLoader by setting shuffle=True
combined_dataset = datasets.ImageFolder(combined_dir, transform=transform)
combined_dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

print("Combined and shuffled dataset created successfully!")
