# Run this file on the ROOT of the project
import numpy as np

# Replace this with the actual path to your embedding file
embedding_path = "./data/mind/small/processed/filtered_embeddings_thresh3.npy"

# If your embedding is a .npy file:
embedding_matrix = np.load(embedding_path)

print("Embedding shape:", embedding_matrix.shape)
print("Min value:", np.min(embedding_matrix))
print("Max value:", np.max(embedding_matrix))
print("Mean value:", np.mean(embedding_matrix))
print("Std deviation:", np.std(embedding_matrix))
print("Any NaNs?:", np.isnan(embedding_matrix).any())
print("Any Infs?:", np.isinf(embedding_matrix).any())