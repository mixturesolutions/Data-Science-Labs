import torch
from sklearn.decomposition import PCA
import numpy as np

# Load extracted features from feature_extracted.pt
features = torch.load('/Users/mixturesolution/Desktop/Data-Science-Labs/Features_Extracted.pt')

# Assuming features are a list of tensors where each tensor represents one batch of data
# First, concatenate all tensors into one large tensor if they're not already
if isinstance(features, list):
    features = torch.cat(features, dim=0)

# Check the shape to ensure it's two-dimensional [samples, features]
print("Shape of features before PCA:", features.shape)

# Convert tensor to a NumPy array for PCA processing
features_np = features.numpy()

# Apply PCA to the 2D array
n_components = 50  # Number of components for dimensionality reduction
pca = PCA(n_components=n_components)
features_reduced = pca.fit_transform(features_np)

# Save the reduced features and the PCA model
torch.save(features_reduced, 'Features_Reduced.pt')
torch.save(pca, 'PCA_model.pth')