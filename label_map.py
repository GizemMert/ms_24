import numpy as np
import matplotlib.pyplot as plt

# Load the confusion matrix
conf_matrix_path = '/home/aih/gizem.mert/ml_24/ms_24/results_deit/cumulative_confusion_matrix.npy'
conf_matrix = np.load(conf_matrix_path, allow_pickle=True)

# Define the label map and class dictionary
label_map = {
    'BAS': 0, 'EBO': 1, 'EOS': 2, 'KSC': 3, 'LYA': 4, 'LYT': 5,
    'MMZ': 6, 'MOB': 7, 'MON': 8, 'MYB': 9, 'MYO': 10, 'NGB': 11,
    'NGS': 12, 'PMB': 13, 'PMO': 14
}

class_dict = {
    'NGS': 'Neutrophil (segmented)',
    'NGB': 'Neutrophil (band)',
    'EOS': 'Eosinophil',
    'BAS': 'Basophil',
    'MON': 'Monocyte',
    'LYT': 'Lymphocyte (typical)',
    'LYA': 'Lymphocyte (atypical)',
    'KSC': 'Smudge Cell',
    'MYO': 'Myeloblast',
    'PMO': 'Promyelocyte',
    'MYB': 'Myelocyte',
    'MMZ': 'Metamyelocyte',
    'MOB': 'Monoblast',
    'EBO': 'Erythroblast',
    'PMB': 'Promyelocyte (bilobed)'
}

# Generate class names based on label_map and class_dict
class_names = [class_dict[key] for key in sorted(label_map.keys(), key=lambda x: label_map[x])]

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
plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)

# Add axis labels and title
plt.xlabel("Network prediction")
plt.ylabel("Examiner label")
plt.title("Confusion Matrix (Relative Frequencies)")

# Tighten layout for better visualization
plt.tight_layout()

# Save the plot
output_path = '/home/aih/gizem.mert/ml_24/ms_24/results_deit/conf_new.png'
plt.savefig(output_path, dpi=300)




