import torch
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from torchvision.datasets import ImageFolder

# Load the extracted test features
test_features = torch.load('Test_Features_Extracted.pt')
if isinstance(test_features, list):
    test_features = torch.cat(test_features, dim=0)

# Check the shape to ensure it's two-dimensional [samples, features]
print("Shape of features before PCA:", test_features.shape)

# Convert tensor directly to a NumPy array
test_features_np = test_features.numpy()

# Load the saved PCA model
pca = torch.load('PCA_model.pth')

# Transform the test features using the loaded PCA
test_features_reduced = pca.transform(test_features_np)

# 'ward' linkage minimizes the variance of clusters being merged.
linked = linkage(test_features_reduced, method='ward')

# Forming flat clusters from the hierarchical clustering
clusters = fcluster(linked, t=525, criterion='maxclust')

# Load the CSV file
csv_path = '/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/birds.csv'
df = pd.read_csv(csv_path, header=None)
print(df.columns.tolist())

# Load image data
dataset = ImageFolder(root='/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/test')

# Create the label mapping
label_mapping = {}
for index, row in df.iterrows():
    if row[3] == 'test':  # Ensure only test paths are included
        # Split the path and extract necessary parts
        path_parts = row[1].split('/')
        if len(path_parts) > 1:
            species_name = path_parts[-2]  # Assuming species name is one level above the file name
            file_name = os.path.basename(row[1])
            # Reform the file_name to match the simple numbering
            file_number = int(file_name.split('.')[0])  # Convert to integer to remove any leading zeros
            file_name = f"{file_number}.jpg"  # Reformat with simple numbering
            # Construct the corrected full path
            full_path = os.path.join('/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/test', species_name, file_name)
            label_mapping[full_path] = row[2]

# Generate test image paths correctly associated with the dataset
test_image_paths = [item[0] for item in dataset.samples]

# Validate and utilize only matched paths
matched_labels = []
clusters_matched = []
for i, path in enumerate(test_image_paths):
    if path in label_mapping:
        matched_labels.append(label_mapping[path])
        clusters_matched.append(clusters[i])
    else:
        print(f"Missing path in label_mapping: {path}")

# Recalculate ARI with matched labels and clusters
ari = adjusted_rand_score(matched_labels, clusters_matched)
print(f"Adjusted Rand Index with matched data: {ari}")

# Visualize clusters
plt.figure(figsize=(10, 8))
plt.scatter(test_features_reduced[:, 0], test_features_reduced[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
plt.title('Test data clustered into groups')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()






