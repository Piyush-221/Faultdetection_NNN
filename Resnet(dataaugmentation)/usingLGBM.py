import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Mounting data
from google.colab import drive
drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/fabric_defects' 

# Geometric data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=data_dir, transform=transform)

# Splitting data into training, validation, and test 
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ResNet
class FabricDefectModel(nn.Module):
    def __init__(self):
        super(FabricDefectModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze ResNet layers
        self.resnet.fc = nn.Identity()  # Remove ResNet's final layer

    def forward(self, x):
        return self.resnet(x)

# Initializing model (to gpu if possible)
model = FabricDefectModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Extracting embeddings from dataset
def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            emb = model(images)
            embeddings.extend(emb.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(embeddings), np.array(labels)

# Embedding extraction for train, validation, and test sets
train_embeddings, train_labels = extract_embeddings(train_loader, model)
val_embeddings, val_labels = extract_embeddings(val_loader, model)
test_embeddings, test_labels = extract_embeddings(test_loader, model)

# Standardization of embeddings
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
val_embeddings = scaler.transform(val_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# Labels -> int
train_labels = train_labels.astype(int)
val_labels = val_labels.astype(int)
test_labels = test_labels.astype(int)

# Creating LGBM datasets
train_data = lgb.Dataset(train_embeddings, label=train_labels)
val_data = lgb.Dataset(val_embeddings, label=val_labels, reference=train_data)

# Define LGBM parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': len(dataset.classes),
    'metric': 'multi_logloss',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'verbose': -1
}

# Train LGBM classifier
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[train_data, val_data], early_stopping_rounds=10)

# Evaluation of classifier on the test set
test_preds = bst.predict(test_embeddings)
test_preds = np.argmax(test_preds, axis=1)
test_accuracy = accuracy_score(test_labels, test_preds)
test_f1_score = f1_score(test_labels, test_preds, average='weighted')

#Accuracy
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test F1 Score: {test_f1_score:.4f}')
