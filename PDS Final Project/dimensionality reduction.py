import torch
from sklearn.decomposition import PCA
import numpy as np

# Step 1: Load extracted features from feature_extracted.pt
features = torch.load('/Users/mixturesolution/Desktop/Data-Science-Labs/Features_Extracted.pt')

# Step 2: Prepare features for PCA
# Find the maximum shape among all tensors
max_shape = max(tensor.shape for tensor in features)

# Pad or truncate tensors to have the same shape
features_padded = [torch.nn.functional.pad(tensor, (0, 0, 0, max_shape[0] - tensor.shape[0])) if tensor.shape[0] < max_shape[0] else tensor[:max_shape[0]] for tensor in features]

# Convert tensors to NumPy arrays
features_np = np.array([tensor.numpy() for tensor in features_padded])

# Reshape the 3D array to a 2D array
n_samples, height, width = features_np.shape
features_2d = features_np.reshape(n_samples, height * width)

# Step 3: Apply PCA to the 2D array
n_components = 50  # Number of components for dimensionality reduction
pca = PCA(n_components=n_components)
features_reduced = pca.fit_transform(features_2d)
torch.save(features_reduced, 'Features_Reduced.pt')
