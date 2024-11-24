import numpy as np
import matplotlib.pyplot as plt

# Load the confusion matrix and check for labels
conf_matrix_path = '/home/aih/gizem.mert/ml_24/ms_24/results_deit/cumulative_confusion_matrix.npy'
conf_matrix = np.load(conf_matrix_path, allow_pickle=True)

# Check if the file contains labels
if isinstance(conf_matrix, dict):  # If the .npy file is stored as a dictionary
    labels = conf_matrix.get('labels', None)  # Extract labels
    conf_matrix = conf_matrix.get('matrix', conf_matrix)  # Extract confusion matrix
else:
    labels = None  # If no labels are found, set to None

# If no labels are found, define them manually
if labels is None:
    labels = [
        "Neutrophil (segmented)", "Neutrophil (band)", "Lymphocyte (typical)", "Lymphocyte (atypical)",
        "Monocyte", "Eosinophil", "Basophil", "Myeloblast", "Promyelocyte",
        "Promyelocyte (bilobed)", "Myelocyte", "Metamyelocyte", "Monoblast",
        "Erythroblast", "Smudge"
    ]

# Normalize the confusion matrix to get relative frequencies
conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_normalized, cmap='Greys', interpolation='nearest')

# Remove grid lines
plt.grid(False)

# Add color bar with explicit ticks
cbar = plt.colorbar(label='Relative frequency')
# cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Explicitly set ticks up to 1.0

# Add labels to axes
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(labels)), labels=labels)

# Add axis labels and title
plt.xlabel("Network prediction")
plt.ylabel("Examiner label")
plt.title("Confusion Matrix (Relative Frequencies)")

# Tighten layout for better visualization
plt.tight_layout()

# Save the plot
output_path = '/home/aih/gizem.mert/ml_24/ms_24/results_deit/conf.png'
plt.savefig(output_path)
plt.show()
