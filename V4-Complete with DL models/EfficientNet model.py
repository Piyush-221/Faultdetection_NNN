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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Mounting data
from google.colab import drive
drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/Fabric_Defect_Dataset'

# Data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(),
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

# EfficientNet with Fine-Tuning
class FabricDefectModel(nn.Module):
    def __init__(self):
        super(FabricDefectModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        for param in self.efficientnet.parameters():
            param.requires_grad = True  # For Fine-tuning the entire model
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 5)  # Adjusting final layer for 5 classes
        self.dropout = nn.Dropout(0.5)  # Adding dropout for regularization

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.dropout(x)
        return x

# Initializing model (to gpu if possible)
model = FabricDefectModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Extracting embeddings from dataset
def extract_embeddings(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
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

# Pipeline
from sklearn.pipeline import Pipeline #Using a pipeline as GridSearchCV tried to clone the entire CaliberatedClassifierCV, including its internal fitted base_estimator
from sklearn.svm import SVC
pipeline = Pipeline([
    ('base_estimator', SVC(kernel='linear', probability=True))
])

# Parameter grid for CalibratedClassifierCV
param_grid = {
    'base_estimator__C': [0.1, 1, 10, 100, 1000],  
}


# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(train_embeddings, train_labels)

# Best parameters
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Train CalibratedClassifierCV with best parameters
best_calibrated_model = CalibratedClassifierCV(SVC(kernel='linear', C=best_params['base_estimator__C'], probability=True), method='sigmoid, cv=3)
best_calibrated_model.fit(train_embeddings, train_labels)

# Evaluation of classifier on the validation and test sets
val_preds = best_calibrated_model.predict(val_embeddings)
val_accuracy = accuracy_score(val_labels, val_preds)
val_f1_score = f1_score(val_labels, val_preds, average='weighted')
val_precision = precision_score(val_labels, val_preds, average='weighted')
val_recall = recall_score(val_labels, val_preds, average='weighted')
val_confusion_matrix = confusion_matrix(val_labels, val_preds)

test_preds = best_calibrated_model.predict(test_embeddings)
test_accuracy = accuracy_score(test_labels, test_preds)
test_f1_score = f1_score(test_labels, test_preds, average='weighted')
test_precision = precision_score(test_labels, test_preds, average='weighted')
test_recall = recall_score(test_labels, test_preds, average='weighted')
test_confusion_matrix = confusion_matrix(test_labels, test_preds)

# Results
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Validation F1 Score: {val_f1_score:.4f}')
print(f'Validation Precision: {val_precision:.4f}')
print(f'Validation Recall: {val_recall:.4f}')
print(f'Validation Confusion Matrix:\n{val_confusion_matrix}')

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test F1 Score: {test_f1_score:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Confusion Matrix:\n{test_confusion_matrix}')
