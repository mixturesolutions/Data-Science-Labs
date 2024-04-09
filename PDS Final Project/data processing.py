import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

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

# load bird image dataset
dataset = ImageFolder(root='/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/train', transform=data_transforms)  # Update with your dataset path
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# extract features from images
features = []
with torch.no_grad():
    # use tqdm to create a progress bar because it takes forever to 
    for images, _ in tqdm(dataloader, desc="Extracting Features", unit="batch", leave=False):
        # Extract features from the second-to-last layer
        outputs = model(images)
        features.append(outputs.squeeze())

# Assuming features is a list of tensors
max_batch_size = max([tensor.size(0) for tensor in features])

# Pad tensors to have consistent batch size
padded_features = [torch.nn.functional.pad(tensor, (0, 0, 0, max_batch_size - tensor.size(0))) for tensor in features]

# Stack padded tensors
stacked_features = torch.stack(padded_features)

# Save extracted features to disk
torch.save(features, 'Features_Extracted.pt')