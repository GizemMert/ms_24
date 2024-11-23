from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import os
from label_map import label_map, class_dict, class_names
import torch

class AML_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, num_folds=5, batch_size=128, random_state=42):

        self.transform = transform
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.random_state = random_state

        self.image_paths = []
        self.labels = []

        # Loop through each folder and assign labels
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in label_map:
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        self.image_paths.append(img_path)
                        self.labels.append(label_map[folder_name])

        # Initialize StratifiedKFold
        self.skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_fold_dataloaders(self):

        fold_dataloaders = []

        for train_idx, test_idx in self.skf.split(self.image_paths, self.labels):
            train_dataset = Subset(self, train_idx)
            test_dataset = Subset(self, test_idx)

            # Further split train into train/val (90% train, 10% val)
            val_split = int(len(train_dataset) * 0.1)
            train_split = len(train_dataset) - val_split
            train_dataset, val_dataset = random_split(
                train_dataset, [train_split, val_split],
                generator=torch.Generator().manual_seed(self.random_state)
            )

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            fold_dataloaders.append((train_loader, val_loader, test_loader))

        return fold_dataloaders



