import torch
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# downloaded path to the pre-trained ResNet-50 weights (download using *wget -c --no-check-certificate https://download.pytorch.org/models/resnet152-b121ed2d.pth*)
weights_path = '/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/resnet50-19c8e357.pth'

# load pre-trained ResNet-50 model without loading pre-trained weights
model = torchvision.models.resnet50(pretrained=False)

# load pre-trained weights using load_state_dict
model.load_state_dict(torch.load(weights_path))

# remove the fully connected layer
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test image dataset with transformations
test_dataset = ImageFolder(root='/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/test', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Extract and flatten features from the test images
test_features = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Extracting Test Features", unit="batch", leave=False):
        outputs = model(images)
        test_features.append(outputs.view(outputs.size(0), -1))

# Concatenate all feature tensors
test_features_concatenated = torch.cat(test_features, dim=0)

# Save the concatenated test features to disk
torch.save(test_features_concatenated, 'Test_Features_Extracted.pt')
