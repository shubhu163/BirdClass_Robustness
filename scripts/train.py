#Importing Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from pathlib import Path
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from sklearn.metrics import classification_report
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# try:
#     import torchinfo
# except:
#     !pip install torchinfo
#     import torchinfo
# from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device is ', device)

TRAIN_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/train"
VALID_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/valid"
TEST_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/test"

#Setting Transfroms
train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Loading Data
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_data = datasets.ImageFolder(VALID_DIR, transform=test_transform) 
test_data = datasets.ImageFolder(TEST_DIR, transform=test_transform)

#Creating Dataloaders
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()
print(NUM_WORKERS)

torch.manual_seed(42)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2 )

class_names = test_data.classes
class_dict = train_data.class_to_idx
test_labels = []
for images, labels in test_dataloader:
    test_labels.extend(labels.tolist())

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
  model.train()

  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    y_train_logits = model(X)

    loss = loss_fn(y_train_logits, y)

    loss.backward()
    optimizer.step()
    train_loss += loss.item()

    y_pred_class = torch.argmax(y_train_logits, dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_train_logits)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc

def valid_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
  model.eval()

  test_loss, test_acc = 0, 0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      y_test_logits = model(X)
      loss = loss_fn(y_test_logits, y)
      test_loss += loss.item()

      y_pred_class = torch.argmax(y_test_logits, dim=1)
      test_acc += (y_pred_class == y).sum().item()/len(y_test_logits)
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,loss_fn: torch.nn.Module):
    model.eval()
        
    correct_count, wrong_count, test_loss, test_acc = 0, 0, 0, 0
    true_labels = []
    predicted_labels = []
    logits = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            true_labels.extend(y.tolist())

            y_test_logits = model(X)
            logits.extend(y_test_logits.tolist())
            loss = loss_fn(y_test_logits, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(y_test_logits, dim=1)
            predicted_labels.extend(y_pred_class.tolist())
            
            correct_count += (y_pred_class == y).sum().item()
            wrong_count += (y_pred_class != y).sum().item()
                
            test_acc += (y_pred_class == y).sum().item()/len(y_test_logits)
            
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return correct_count, wrong_count, logits, predicted_labels, true_labels, test_loss, test_acc

def train_loop(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               valid_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module = nn.CrossEntropyLoss(),
               epochs: int = 10,
               patience: int = 5):
  results = {
      "train_loss": [],
      "train_acc": [],
      "valid_loss": [],
      "valid_acc": []
  }
  best_val_loss = float('inf')
  counter = 0

  for epoch in tqdm(range(epochs)):
    final_train_loss, final_train_acc = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer)

    final_valid_loss, final_valid_acc = valid_step(
        model=model,
        dataloader=valid_dataloader,
        loss_fn=loss_fn
    )
    
    print(f"Epoch: {epoch+1} | "
          f"train loss: {final_train_loss:.4f} | "
          f"train accuracy: {final_train_acc:.4f} | "
          f"valid loss: {final_valid_loss:.4f} | "
          f"valid accuracy: {final_valid_acc:.4f}"
          )

    results["train_loss"].append(final_train_loss)
    results["train_acc"].append(final_train_acc)
    results["valid_loss"].append(final_valid_loss)
    results["valid_acc"].append(final_valid_acc)
    
    if final_valid_loss < best_val_loss:
        best_val_loss = final_valid_loss
        counter=0
        torch.save(model.state_dict(), "fine_best_model_birds.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Stopping...")
            break
  return results

#Training Begins Now
EPOCHS = 30
weights = Wide_ResNet50_2_Weights.DEFAULT
model = wide_resnet50_2(weights=weights)

for param in model.parameters():
    param.requires_grad = True

# Modify the fully connected layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 525),  # Adjust to your number of species
    nn.LogSoftmax(dim=1)
)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


model_1_results = train_loop(model=model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,optimizer=optimizer, loss_fn=loss_fn, epochs=EPOCHS,patience=5)

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "valid_loss": [...],
             "valid_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    valid_loss = results['valid_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    valid_accuracy = results['valid_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, valid_loss, label='valid_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.savefig('Loss.png')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, valid_accuracy, label='valid_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend
    plt.savefig('Accuracy.png')

plot_loss_curves(model_1_results)
