import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),  # Adjusted size to fit InceptionV3 input
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(299),  # Adjusted size to fit InceptionV3 input
        transforms.CenterCrop(299),  # Adjusted size to fit InceptionV3 input
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(299),  # Adjusted size to fit InceptionV3 input
        transforms.CenterCrop(299),  # Adjusted size to fit InceptionV3 input
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Data directories
data_dir = '/nfs/hpc/share/joshishu/DL_Project/data'
batch_size = 32
# Load datasets
# # count = 0
# # for file in os.listdir(os.path.join(data_dir, 'train')):
# #     print(file)
# #     count +=1 
# # print(count)

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
# print(len(train_dataset))
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transforms['valid'])
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define the base model (InceptionV3)
def get_base_model():
    model = models.inception_v3(pretrained=True)
    # Modify the final fully connected layer to match the number of classes in your dataset
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, 525)  # Adjusted to match the number of classes
    model = model.to(device)
    return model

# Training function
def train_model(model, optimizer, criterion, num_epochs=25):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # Use only the main outputs

            # Extract logits from outputs
            logits_train = outputs.logits

            # Calculate loss
            loss = criterion(logits_train, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits_train, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Acc: {epoch_acc:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # Use only the main outputs
            logits_val = outputs.logits

            loss = criterion(logits_val, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits_val, 1)
            val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Testing
    model.eval()
    test_corrects = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # Use only the main outputs
        logits_test = outputs.logits

        _, preds = torch.max(logits_test, 1)
        test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double() / len(test_dataset)
    print(f'Test Accuracy: {test_acc:.4f}')


# Train the model
base_model = get_base_model()
optimizer = optim.Adam(base_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_model(base_model, optimizer, criterion)



# Train the model
base_model = get_base_model()
optimizer = optim.Adam(base_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_model(base_model, optimizer, criterion)