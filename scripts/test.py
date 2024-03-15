import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from sklearn.metrics import classification_report
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
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
  
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device is ', device)

TEST_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/test"

#Setting Transfroms

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Loading Data
test_data = datasets.ImageFolder(TEST_DIR, transform=test_transform)

#Creating Dataloaders
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()
print(NUM_WORKERS)

torch.manual_seed(42)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2 )

# Assuming you fine-tuned the model and saved it
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
# print(checkpoint.keys())


loss_fn = nn.CrossEntropyLoss()

correct_count, wrong_count, logits, predicted_labels, true_labels, test_loss, test_acc = test_step(
    model=model,
    dataloader=test_dataloader,
    loss_fn=loss_fn)

report = classification_report(true_labels, predicted_labels, output_dict=True)

print(f"Predicted Label:\n{test_data.classes[predicted_labels[7]]}\n")
print(f"Actual Label:\n{test_data.classes[true_labels[7]]}\n")
print(f"Loss: {test_loss}")
print(f"AVG Accuracy: {report['accuracy']:.4f}\n")

print(f"Macro AVG Precision: {report['macro avg']['precision']:.4f}")
print(f"Macro AVG Recall: {report['macro avg']['recall']:.4f}")
print(f"Macro AVG F1: {report['macro avg']['f1-score']:.4f}")

print(f"Weighted AVG Precision: {report['weighted avg']['precision']:.4f}")
print(f"Weighted AVG Recall: {report['weighted avg']['recall']:.4f}")
print(f"Weighted AVG F1: {report['weighted avg']['f1-score']:.4f}")

print(f"Total Predictions:{len(predicted_labels)}")
print(f"Total Labels:{len(true_labels)}")
print(f"Predictions 0 - 10:\n{predicted_labels[:20]}")
print(f"Labels 0 - 10:\n{true_labels[:20]}")
print(f"Correct:{correct_count}")
print(f"Wrong:{wrong_count}")
