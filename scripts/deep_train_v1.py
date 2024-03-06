#!/usr/bin/env python
# coding: utf-8

# # Intento de red neuronal usando el datased Birds 525 No.1
# 

# ## 1: Importar librerias y configurer el API de Kaggle

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from pathlib import Path
import zipfile
import os
from sklearn.metrics import classification_report
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# try:
#     import torchinfo
# except:
#     get_ipython().system('pip install torchinfo')
#     import torchinfo
# from torchinfo import summary


# In[2]:


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# ## 2: Preparar el dataset (crear estructura de carpetas y descomprimir)
# 

# In[3]:


#DATA_PATH = Path("data/")
#IMAGE_PATH = DATA_PATH / "birds"

#if IMAGE_PATH.is_dir():
#  print(f"El directorio {IMAGE_PATH} ya existe.")
#else:
#  print(f"El directorio {IMAGE_PATH} no ha sido encontrado, creandolo...")
#  IMAGE_PATH.mkdir(parents=True, exist_ok=True)
#  print(f"El directorio {IMAGE_PATH} ha sido creado exitosamente!")

#  with zipfile.ZipFile("100-bird-species.zip", "r") as zip_dataset:
#    print(f"Descomprimiendo el dataset...")
#    zip_dataset.extractall(IMAGE_PATH)
#    print(f"Dataset Descomprimido Exitosamente...")


# ## 3: Transformar y cargar el dataset

# In[4]:


TRAIN_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/train"
VALID_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/valid"
TEST_DIR = "/nfs/hpc/share/joshishu/DL_Project/data/test"
TRAIN_DIR, VALID_DIR, TEST_DIR


# In[5]:


# crear el transform
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


# In[6]:


# transformar las carpetas a datasets
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_data = datasets.ImageFolder(VALID_DIR, transform=test_transform) 
test_data = datasets.ImageFolder(TEST_DIR, transform=test_transform)
train_data, valid_data, test_data


# In[7]:


# crear los dataloader

BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)
train_dataloader = DataLoader(train_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=2
                              )

valid_dataloader = DataLoader(valid_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=2
                             )

test_dataloader = DataLoader(test_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=2
                             )
train_dataloader, valid_dataloader, test_dataloader


# In[8]:


class_names = test_data.classes

print(len(class_names))
class_dict = train_data.class_to_idx
print(len(class_dict))

test_labels = []
for images, labels in test_dataloader:
    test_labels.extend(labels.tolist())
print(len(test_labels))


# ## 4: Construir el modelo

# In[9]:


# class SurfinBird(nn.Module):
#   def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
#     super().__init__()
#     self.conv1 = nn.Conv2d(
#             in_channels=input_shape,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3)
#     self.bn1 = nn.BatchNorm2d(64)
#     self.relu1 = nn.ReLU()
#     self.mp1 = nn.MaxPool2d(kernel_size=2,
#                      stride=2)
#     self.conv_block_2 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2,
#                      stride=2)
#     )
#     self.conv_block_3 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=64,
#             out_channels=128,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=128,
#             out_channels=128,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=128,
#             out_channels=128,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2,
#                      stride=2)
#     )
#     self.conv_block_4 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=128,
#             out_channels=128,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=128,
#             out_channels=128,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=128,
#             out_channels=128,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2,
#                      stride=2)
#     )
#     self.conv_block_5 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=128,
#             out_channels=256,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(256),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(256),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(256),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2,
#                      stride=2)
#     )
#     self.conv_block_6 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(256),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(256),
#         nn.ReLU(),
#         nn.Conv2d(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         ),
#         nn.BatchNorm2d(256),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2,
#                      stride=2)
#     )
    
#     self.avgpool = nn.Sequential(
#     nn.AdaptiveAvgPool2d(output_size=(1, 1))
#     )
    
#     self.classifier = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(in_features=hidden_units*1*1,
#                   out_features=output_shape)
#     )
#   def forward(self, x: torch.Tensor):
#     return self.classifier(self.avgpool(self.conv_block_6(self.conv_block_5(self.conv_block_4(self.conv_block_3(self.conv_block_2(self.mp1(self.relu1(self.bn1(self.conv1(x)))))))))))

#torch.manual_seed(42)
#model_0 = SurfinBird(input_shape=3,
#                     hidden_units=10,
#                     output_shape=len(train_data.classes)).to(device)
#model_0


# In[10]:


#summary(model_0, input_size=[1, 3, 128, 128])


# ### 4.1: Una pequeña prueba con el modelo

# In[11]:


img_batch, label_batch = next(iter(train_dataloader))
shapes = []
print(img_batch.shape)
print(train_dataloader.batch_size)
for img in img_batch:
  shapes.append(img.shape)
print(set(shapes))
#print(shapes)

test_img_batch, test_label_batch = next(iter(test_dataloader))
test_shapes = []
for test_img in test_img_batch:
  test_shapes.append(test_img.shape)
print(set(test_shapes))


# In[12]:


#img_batch, label_batch = next(iter(train_dataloader))
#single_img, single_label = img_batch[0].unsqueeze(dim=0), label_batch[0]
#print(f"forma de la imagen: {single_img.shape}")

#model_0.eval()
#with torch.inference_mode():
#  pred = model_0(single_img.to(device))

#print(f"logits:\n{pred}\n")
#print(f"Probabilidades:\n{torch.softmax(pred, dim=1)}\n")
#print(f"Etiqueta predicha:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
#print(f"Etiqueta actual:\n{single_label}")


# ## 5: Crear las funciones de test, train y el bucle
# 

# In[13]:


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


# In[14]:


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


# In[15]:


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


# ### 5.1: bucle bucle bucle..... viril

# In[16]:


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
        torch.save(model.state_dict(), "best_model_birds.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Stopping...")
            break
  return results


# ## 6: DJ train dat shit!!!

# In[17]:


#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
EPOCHS = 100

# model_1 = SurfinBird(input_shape=3,
#                      hidden_units=256,
#                      output_shape=len(train_data.classes)).to(device)
weights = Wide_ResNet50_2_Weights.DEFAULT
model = wide_resnet50_2(weights=weights)
# model = models.wide_resnet50_2(pretrained=True)

# Freeze parameters so we don't backprop through them
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
# summary(model_1, input_size=[1, 3, 224, 224])


# In[ ]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


model_1_results = train_loop(model=model,
                             train_dataloader=train_dataloader,
                             valid_dataloader=valid_dataloader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             epochs=EPOCHS,
                             patience=5)


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Tuple, Dict, List
# def plot_loss_curves(results: Dict[str, List[float]]):
#     """Plots training curves of a results dictionary.

#     Args:
#         results (dict): dictionary containing list of values, e.g.
#             {"train_loss": [...],
#              "train_acc": [...],
#              "valid_loss": [...],
#              "valid_acc": [...]}
#     """

#     # Get the loss values of the results dictionary (training and test)
#     loss = results['train_loss']
#     valid_loss = results['valid_loss']

#     # Get the accuracy values of the results dictionary (training and test)
#     accuracy = results['train_acc']
#     valid_accuracy = results['valid_acc']

#     # Figure out how many epochs there were
#     epochs = range(len(results['train_loss']))

#     # Setup a plot
#     plt.figure(figsize=(15, 7))

#     # Plot loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, loss, label='train_loss')
#     plt.plot(epochs, valid_loss, label='valid_loss')
#     plt.title('Loss')
#     plt.xlabel('Epochs')
#     plt.legend()

#     # Plot accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, accuracy, label='train_accuracy')
#     plt.plot(epochs, valid_accuracy, label='valid_accuracy')
#     plt.title('Accuracy')
#     plt.xlabel('Epochs')
#     plt.legend();


# # In[ ]:


# plot_loss_curves(model_1_results)


# # ## 7: Test the model (test set)
# # 

# # In[ ]:


# loaded_best_birds = SurfinBird(input_shape=3,
#                      hidden_units=256,
#                      output_shape=len(train_data.classes))
# loaded_best_birds.load_state_dict(torch.load(f="best_model_birds.pth"))
# loaded_best_birds = loaded_best_birds.to(device)


# # In[ ]:


# correct_count, wrong_count, logits, predicted_labels, true_labels, test_loss, test_acc = test_step(
#     model=loaded_best_birds,
#     dataloader=test_dataloader,
#     loss_fn=loss_fn)

# report = classification_report(true_labels, predicted_labels, output_dict=True)

# print(f"Etiqueta predicha:\n{test_data.classes[predicted_labels[7]]}\n")
# print(f"Etiqueta actual:\n{test_data.classes[true_labels[7]]}\n")
# print(f"Loss: {test_loss}")
# print(f"AVG Accuracy: {report['accuracy']:.4f}\n")

# print(f"Macro AVG Precision: {report['macro avg']['precision']:.4f}")
# print(f"Macro AVG Recall: {report['macro avg']['recall']:.4f}")
# print(f"Macro AVG F1: {report['macro avg']['f1-score']:.4f}")

# print(f"weighted AVG Precision: {report['weighted avg']['precision']:.4f}")
# print(f"weighted AVG Recall: {report['weighted avg']['recall']:.4f}")
# print(f"weighted AVG F1: {report['weighted avg']['f1-score']:.4f}")

# print(f"Predicciones total:{len(predicted_labels)}")
# print(f"Etiquetas total:{len(true_labels)}")
# print(f"Predicciones 0 - 10:\n{predicted_labels[:20]}")
# print(f"Etiquetas 0 - 10:\n{true_labels[:20]}")
# print(f"Correctas:{correct_count}")
# print(f"Erroneas:{wrong_count}")


# # ## 8: Predict user generated images

# # In[ ]:


# #WIP

