import os
from collections import defaultdict

# Define the dataset path
dataset_path = "/lustre/groups/aih/raheleh.salehi/Datasets/Cytomorphology_Matek/AML-Cytomorphology_LMU"

# Dictionary to store the count of images in each class
class_image_count = defaultdict(int)

# Loop through each folder in the dataset
for class_folder in os.listdir(dataset_path):
    class_folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_folder_path):
        # Count the number of .tiff images in the folder
        image_count = len([f for f in os.listdir(class_folder_path) if f.endswith(".tiff")])
        class_image_count[class_folder] = image_count

# Print the count of images in each class
for class_name, count in class_image_count.items():
    print(f"Class: {class_name}, Number of Images: {count}")
