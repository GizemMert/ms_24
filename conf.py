import numpy as np

# Load the .npy file
conf_matrix_path = '/home/aih/gizem.mert/ml_24/ms_24/results_deit/cumulative_confusion_matrix.npy'
data = np.load(conf_matrix_path, allow_pickle=True)

# Check the type of data
print("Type of data:", type(data))

# If it's a dictionary, list the keys
if isinstance(data, dict):
    print("Keys in the .npy file:", data.keys())

# If it's not a dictionary, display the shape or content
else:
    print("Shape of data:", data.shape)
    print("Data preview:")
    print(data)
