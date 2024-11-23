import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import os
import json
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from model_vit import DeiTTinyClassifier
from label_map import class_names
from label_map import label_map, class_dict, class_names


# Define Human-Readable Class Names
def train_deit_tiny(
    fold_dataloaders, class_names, num_epochs=150, patience=20, output_dir="results_deit"
):
    """
    Train a DeiT-Tiny model using 5-fold cross-validation with early stopping.

    Args:
        model (torch.nn.Module): PyTorch model to train.
        fold_dataloaders (list): List of train, val, and test dataloaders for each fold.
        class_names (list): List of human-readable class names.
        num_epochs (int): Maximum number of epochs.
        patience (int): Number of epochs to wait for validation improvement before stopping.
        output_dir (str): Directory to save logs, metrics, and model checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeiTTinyClassifier(num_classes=len(class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_logs_deit.txt")

    # Initialize storage for cumulative confusion matrix and metrics
    cumulative_confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    all_metrics = []  # Store metrics for all folds

    for fold, (train_loader, val_loader, test_loader) in enumerate(fold_dataloaders):
        print(f"\n=== Starting Fold {fold + 1}/{len(fold_dataloaders)} ===\n")
        wandb.init(
            project="DeiT-Tiny-5Fold",
            name=f"Fold-{fold + 1}",
            config={
                "num_epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": 0.001,
                "architecture": "DeiT-Tiny",
                "num_folds": len(fold_dataloaders),
                "early_stopping_patience": patience
            },
            reinit=True
        )
        wandb.log({"fold": fold + 1})
        print(
            f"Train: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        wandb.log({"fold": fold + 1})
        with open(log_file, "a") as f:
            f.write(f"\n=== Fold {fold + 1}/{len(fold_dataloaders)} ===\n")

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            avg_train_loss = train_loss / total

            # Validation phase
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            avg_val_loss = val_loss / total

            # Log to wandb
            wandb.log({
                f"Fold {fold + 1}": {
                    "Epoch": epoch + 1,
                    "Train Loss": avg_train_loss,
                    "Train Accuracy": train_acc,
                    "Validation Loss": avg_val_loss,
                    "Validation Accuracy": val_acc
                }
            })

            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Save the best model for the current fold
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, os.path.join(output_dir, f"best_deit_model_fold{fold + 1}.pth"))

        # Evaluate on the test set
        print(f"\n=== Testing Fold {fold + 1} ===")
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Confusion matrix for the fold
        fold_cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))
        cumulative_confusion_matrix += fold_cm

        # Class-wise metrics
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=np.arange(len(class_names)), zero_division=0
        )
        fold_accuracy = np.trace(fold_cm) / np.sum(fold_cm)
        fold_metrics = {
            "class_metrics": {
                class_names[i]: {
                    "precision": precision[i],
                    "recall (sensitivity)": recall[i],
                    "f1_score": f1_score[i]
                }
                for i in range(len(class_names))
            },
            "overall_accuracy": fold_accuracy,
        }

        with open(os.path.join(output_dir, f"fold_{fold + 1}_metrics.json"), "w") as f:
            json.dump(fold_metrics, f)

        all_metrics.append(fold_metrics)
        wandb.finish()

    # Save cumulative confusion matrix and metrics
    np.save(os.path.join(output_dir, "cumulative_confusion_matrix.npy"), cumulative_confusion_matrix)
    save_confusion_matrix(
        cumulative_confusion_matrix,
        class_names,
        os.path.join(output_dir, "cumulative_confusion_matrix.png")
    )

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    print("\nTraining completed. Results saved to:", output_dir)


def save_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Cumulative Confusion Matrix")
    plt.savefig(output_path)
    plt.close()
