import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(42)
n_samples = 100
n_features = 2
X = np.random.rand(n_samples, n_features) * 10

# Initialize and fit the PCA model
pca = PCA(n_components=2)
pca.fit(X)

# Transform the data to the principal components
X_pca = pca.transform(X)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the original data and the transformed data
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

print("Explained Variance Ratio:", explained_variance_ratio)