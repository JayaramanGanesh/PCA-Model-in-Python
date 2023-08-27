import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

# Generate a synthetic dataset with noise
X, _ = make_moons(n_samples=200, noise=0.2, random_state=42)

# Apply PCA for noise reduction
n_components = 1  # Number of principal components to retain
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Reconstruct the data using inverse transform
X_reconstructed = pca.inverse_transform(X_pca)

# Plot the original data and the reconstructed data
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
plt.title('Original Noisy Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.8)
plt.title('Reconstructed Data (PCA)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()