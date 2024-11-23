from model_vit import DeiTTinyClassifier
from train import train_deit_tiny
import torchvision.transforms as transforms
from dataset import AML_Dataset
from label_map import label_map, class_dict
class_names = [class_dict[key] for key in sorted(label_map.keys(), key=lambda x: label_map[x])]



transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT input
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset_path = "/lustre/groups/aih/raheleh.salehi/Datasets/Cytomorphology_Matek/AML-Cytomorphology_LMU"
dataset = AML_Dataset(root_dir=dataset_path, transform=transform, num_folds=5, batch_size=128)
output_dir = "results_deit"  # Directory to save all logs and metrics
batch_size = 128
num_folds = 5
num_epochs = 100
patience = 20

# Load Data
fold_dataloaders = dataset.get_fold_dataloaders()
# Initialize the DeiT-Tiny Model
num_classes = len(class_names)
model = DeiTTinyClassifier(num_classes=num_classes)

# Train the Model with Early Stopping
train_deit_tiny(
    model,
    fold_dataloaders,
    class_names,
    num_epochs=num_epochs,
    patience=patience,
    output_dir=output_dir  # Set the output directory
)
