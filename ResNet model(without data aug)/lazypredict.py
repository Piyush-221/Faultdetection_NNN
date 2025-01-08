import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
!pip install lazypredict
from lazypredict.Supervised import LazyClassifier
import numpy as np
from tqdm import tqdm
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Dataset and Transformations
data_dir = '/content/drive/MyDrive/Fabric_Defect_Dataset'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=data_dir, transform=transform)

# Spliting the dataset
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ResNet-based Model
class FabricDefectModel(nn.Module):
    def __init__(self, num_classes):
        super(FabricDefectModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final layer
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)


num_classes = len(dataset.classes)
model = FabricDefectModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to Extract Features
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, target_labels in tqdm(dataloader):
            embeddings = model.resnet(images)
            features.append(embeddings.cpu().numpy())
            labels.extend(target_labels.cpu().numpy())
    return np.vstack(features), np.array(labels)

# Extract Features
train_features, train_labels = extract_features(model, train_loader)
val_features, val_labels = extract_features(model, val_loader)
test_features, test_labels = extract_features(model, test_loader)

# Combine Train and Validation Features
X = np.vstack([train_features, val_features])
y = np.hstack([train_labels, val_labels])

# Use Test Features Separately
X_test = test_features
y_test = test_labels

# Scale the Features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Split Data for Lazy Predict
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Lazy Predict
lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazy_clf.fit(X_train, X_val, y_train, y_val)
print("Model Evaluation Metrics:")
print(models)

# Evaluate the Best Model
from sklearn.ensemble import RandomForestClassifier

best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

# Evaluate on the Test Set
test_accuracy = best_model.score(X_test, y_test)
test_f1_score = f1_score(y_test, best_model.predict(X_test), average='weighted')

print(f"Test Accuracy (Lazy Predict Best Model): {test_accuracy:.4f}")
print(f"Test F1 Score (Lazy Predict Best Model): {test_f1_score:.4f}")
