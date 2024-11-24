import numpy as np
import matplotlib.pyplot as plt

# Load the confusion matrix from the npy file
conf_matrix = np.load('/home/aih/gizem.mert/ml_24/ms_24/results_deit/cumulative_confusion_matrix.npy')  # Replace with your file path
output_path = '/home/aih/gizem.mert/ml_24/ms_24/results_deit/conf.png'
# Normalize the confusion matrix to get relative frequencies
conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Define labels
labels = [
    "Neutrophil (segmented)", "Neutrophil (band)", "Lymphocyte (typical)", "Lymphocyte (atypical)",
    "Monocyte", "Eosinophil", "Basophil", "Myeloblast", "Promyelocyte",
    "Promyelocyte (bilobed)", "Myelocyte", "Metamyelocyte", "Monoblast",
    "Erythroblast", "Smudge"
]

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_normalized, cmap='Greys', interpolation='nearest')

# Remove grid lines
plt.grid(False)

# Add color bar with explicit ticks
cbar = plt.colorbar(label='Relative frequency')
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Explicitly set ticks up to 1.0

# Add labels to axes
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(labels)), labels=labels)

# Add axis labels and title
plt.xlabel("Network prediction")
plt.ylabel("Examiner label")
plt.title("Confusion Matrix (Relative Frequencies)")

# Tighten layout for better visualization
plt.tight_layout()

plt.savefig(output_path)

