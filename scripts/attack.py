import torch
import torch.nn as nn
import torchattacks
from torchvision import models, transforms
from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

TEST_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/test"

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder(TEST_DIR, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False, num_workers=2)


# Load your trained model
# For demonstration, I'm using a pretrained ResNet18 model
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
model.eval()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define the attacks
epsilon = 0.03  # Perturbation level
fgsm_attack = torchattacks.FGSM(model, eps=epsilon)
pgd_attack = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)
bim_attack = torchattacks.BIM(model, eps=epsilon, alpha=2/255, steps=10)

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
