import numpy as np
from sklearn.decomposition import PCA

# Generate a synthetic dataset
np.random.seed(42)
n_samples = 100
n_features = 5
X = np.random.rand(n_samples, n_features) * 10

# Initialize and fit the PCA model
n_components = 3  # Number of principal components to retain
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Display the original variance and explained variance ratio
print("Original Variance:", np.var(X, axis=0))
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Display the transformed data
print("Transformed Data (First 5 samples):\n", X_pca[:5, :])