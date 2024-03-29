import torch
import torch.nn as nn
import torchattacks
from torchvision import models, transforms
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

DATA_DIR = '/nfs/hpc/share/joshishu/DL_Project/data/test'

# Define the transform
all_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load all data
all_data = datasets.ImageFolder(DATA_DIR, transform=all_transform)

# Calculate split sizes for training, validation, and test sets
total_size = len(all_data)
train_size = int(0.7 * total_size)  # 70% of data for training
valid_size = int(0.15 * total_size)  # 15% of data for validation
test_size = total_size - train_size - valid_size  # The rest for testing



torch.manual_seed(42)  # For reproducibility
train_data, valid_data, test_data = random_split(all_data, [train_size, valid_size, test_size])
train_dataloader = DataLoader(all_data, batch_size=32, shuffle=False, num_workers=10)




weights = Wide_ResNet50_2_Weights.DEFAULT
model = wide_resnet50_2()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 525),  # Adjust to your number of species
    nn.LogSoftmax(dim=1)
)

model.load_state_dict(torch.load(f="/nfs/hpc/share/joshishu/DL_Project/BirdClass_Robustness/Adversarial_Wide/SUPER_MODEL/WideResnet_ADV_Shuffled_14.pth"))
model.eval()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define the attacks
epsilon = 0.03  # Perturbation level
fgsm_attack = torchattacks.FGSM(model, eps=epsilon)
pgd_attack = torchattacks.PGD(model, eps=epsilon, alpha=1/255, steps=4)
bim_attack = torchattacks.BIM(model, eps=epsilon, alpha=1/255, steps=4)

def attack_model(model, dataloader, attack):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Generate adversarial images
        adv_images = attack(X, y)

        # Forward pass with adversarial images
        y_pred_logits = model(adv_images)

        # Compute accuracy
        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        correct += (y_pred_class == y).sum().item()
        total += y.size(0)

    accuracy = 100 * correct / total
    print(f'Accuracy under {attack} : {accuracy}%')


attack_model(model, train_dataloader, fgsm_attack)
attack_model(model, train_dataloader, pgd_attack)
attack_model(model, train_dataloader, bim_attack)
