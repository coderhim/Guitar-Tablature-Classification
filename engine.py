import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import GuitarTabNet
import my_dataloader
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from tqdm import tqdm
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, save_path='best_guitar_tab_model.pt'):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    string_accuracies = [[] for _ in range(6)]

    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)

            # Convert one-hot encoded labels to class indices
            target_indices = []
            for label in labels:
                if label.dim() > 1 and label.shape[1] > 1:
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    indices = label.view(-1).to(device).long()
                target_indices.append(indices)

            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1)  # Ensure input shape is consistent
            outputs = model(inputs)

            # Compute total loss across all outputs
            loss = sum(criterion(output, target) for output, target in zip(outputs, target_indices))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

        # Evaluate on validation set
        val_loss, accuracies, preds, targets = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)

        for i, acc in enumerate(accuracies):
            string_accuracies[i].append(acc)

        # Collect predictions and targets for confusion matrices
        for i in range(6):
            all_preds[i].extend(preds[i])
            all_targets[i].extend(targets[i])

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with validation loss: {val_loss:.4f}")

        # Update training plots
        plot_training_metrics(train_losses, val_losses, string_accuracies)

    # Plot confusion matrices after training
    plot_confusion_matrices(all_preds, all_targets)

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = [0] * 6
    total = 0

    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            target_indices = []
            for label in labels:
                if label.dim() > 1 and label.shape[1] > 1:
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    indices = label.view(-1).to(device).long()
                target_indices.append(indices)

            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)

            batch_loss = sum(criterion(output, target) for output, target in zip(outputs, target_indices))
            val_loss += batch_loss.item()

            for i in range(6):
                pred = torch.argmax(outputs[i], dim=1)
                correct[i] += (pred == target_indices[i]).sum().item()

                all_preds[i].extend(pred.cpu().numpy())
                all_targets[i].extend(target_indices[i].cpu().numpy())

            total += target_indices[0].size(0)

    accuracies = [c / total * 100 for c in correct]

    print(f"Validation Loss: {val_loss:.4f}")
    for i, acc in enumerate(accuracies):
        print(f"Accuracy for string {i + 1}: {acc:.2f}%")

    return val_loss, accuracies, all_preds, all_targets

def plot_training_metrics(train_losses, val_losses, string_accuracies):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(2, 1, 2)
    for i, accs in enumerate(string_accuracies):
        plt.plot(accs, label=f'String {i+1}')
    plt.title('Validation Accuracy by String')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_confusion_matrices(all_preds, all_targets):
    """
    Plot confusion matrices for each string (6 classes) and display them inline in Google Colab.

    Parameters:
    - all_preds: List of predictions for each string (length 6).
    - all_targets: List of true labels for each string (length 6).
    """
    plt.figure(figsize=(20, 15))

    for i in range(6):
        if len(all_preds[i]) > 0:
            cm = confusion_matrix(all_targets[i], all_preds[i])
            plt.subplot(2, 3, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - String {i + 1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')

    plt.tight_layout()
    plt.savefig('/content/confusion_matrices.png')
    print("Confusion matrices saved as 'confusion_matrices.png'")
    plt.close()


def test_model(model, test_loader, device):
    model.eval()
    correct = [0] * 6
    total = [0] * 6

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            # Ensure correct input shape
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)

            # Process labels
            target_indices = []
            for label in labels:
                if label.dim() > 1 and label.shape[1] > 1:
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    indices = label.to(device).long()
                target_indices.append(indices)

            # Evaluate predictions
            for i, (output, target) in enumerate(zip(outputs, target_indices)):
                _, predicted = torch.max(output, 1)
                correct[i] += (predicted == target).sum().item()
                total[i] += target.size(0)

    # Report accuracy
    for i in range(6):
        accuracy = 100 * correct[i] / total[i] if total[i] > 0 else 0
        print(f"Test Accuracy for string {i + 1}: {accuracy:.2f}%")

if __name__ == '__main__':
    # Load data
    image_dir = r'/content/Guitar-Tablature-Classification/cqt_audio'
    annotation_dir = r'/content/Guitar-Tablature-Classification/tablature_segments'
    set_seed()
    train_loader, val_loader, test_loader = my_dataloader.create_dataloaders(audio_dir, annotation_dir)

    # Inspect a sample batch
    for audio, heads in train_loader:
        print("Audio shape:", audio.shape)
        print("Head shapes:", [h.shape for h in heads])
        break
    # Hyperparameters
    input_shape = (1, 96, 9)  # Channel, Height, Width
    epochs = 30
    learning_rate = 0.01
    momentum = 0.8
    decay = learning_rate / epochs
    
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GuitarTabNet(input_shape).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    criterion = nn.CrossEntropyLoss()
    # Start training
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

    print("Testing the model")
     # Testing
    print("Testing the trained model now")
    model.eval()
    model.to(device)
    test_model(model, test_loader, device)
