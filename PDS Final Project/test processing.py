import torch
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load pre-extracted features for the training data
train_features = torch.load('Features_Reduced.pt')

# Assuming you have your cluster labels as a NumPy array
train_labels = np.loadtxt('Cluster_Labels.txt')

# Initialize and train the classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_features, train_labels)

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
dataset = ImageFolder(root='/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/test', transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Extract features from the test images
test_features = []
with torch.no_grad():
    for images, _ in tqdm(dataloader, desc="Extracting Test Features", unit="batch", leave=False):
        # Extract features from the second-to-last layer
        outputs = model(images)
        test_features.append(outputs.squeeze())

# Handle possible variable tensor sizes in the test set as you did with the training set
test_max_batch_size = max([tensor.size(0) for tensor in test_features])

# Pad tensors to have consistent batch size
test_padded_features = [torch.nn.functional.pad(tensor, (0, 0, 0, test_max_batch_size - tensor.size(0))) for tensor in test_features]

# Stack padded tensors
test_stacked_features = torch.stack(test_padded_features)

# Save extracted test features to disk
torch.save(test_stacked_features, 'Test_Features_Extracted.pt')

# Convert the list of tensors to a single tensor first
test_features_tensor = torch.cat(test_features)

# Predict cluster labels for all test features
test_labels_pred = classifier.predict(test_features_tensor.numpy())

# Get the true labels from the dataset
true_labels = np.array([label for _, label in dataset.samples])

# Calculate the percentage of correct predictions
accuracy = np.sum(test_labels_pred == true_labels) / len(true_labels)
print(f"Accuracy of categorization: {accuracy * 100:.2f}%")

# Output the predicted vs true categories for each image (optional)
for i, (true_label, predicted_label) in enumerate(zip(true_labels, test_labels_pred)):
    print(f"Image {i}: True Cluster {true_label}, Predicted Cluster {predicted_label}")
