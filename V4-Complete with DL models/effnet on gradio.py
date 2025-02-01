import gradio as gr
import torch
from PIL import Image
import numpy as np
import cv2

# Load your trained model
class FabricDefectModel(nn.Module):
    def __init__(self):
        super(FabricDefectModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.dropout(x)
        return x

model = FabricDefectModel()
model.load_state_dict(torch.load('path_to_your_model.pth'))  # Load your trained model
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the prediction function
def predict(image):
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    label = predicted.item()
    
    # Assuming 'label_names' is a list of your label names
    label_names = ['Defect1', 'Defect2', 'Defect3', 'Defect4', 'Defect5']
    
    # Draw bounding box (example, you can modify as needed)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), 2)
    cv2.putText(image, label_names[label], (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return image

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(224, 224)),
    outputs="image"
)

# Launch the interface
interface.launch(debug=True)
