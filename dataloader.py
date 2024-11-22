import os
from collections import defaultdict

# Define the dataset path
dataset_path = "/lustre/groups/aih/raheleh.salehi/Datasets/Cytomorphology_Matek/AML-Cytomorphology_LMU"

# List to store problematic files
double_extension_files = []

# Loop through all files in the dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".tiff.tiff"):  # Check for double .tiff extensions
            double_extension_files.append(os.path.join(root, file))

# Print results
if double_extension_files:
    print("Files with .tiff.tiff extensions found:")
    for file in double_extension_files:
        print(file)
else:
    print("No files with .tiff.tiff extensions found.")
