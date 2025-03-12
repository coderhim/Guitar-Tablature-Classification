import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from tqdm import tqdm
import my_dataloader
# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def time_shift(audio, shift_range=0.1):
    """Randomly shift audio in time"""
    time_dim = audio.shape[2]
    if time_dim < 2:  
        return audio  # Skip if time dimension is too small
    
    shift = int(random.uniform(-shift_range, shift_range) * time_dim)
    if shift > 0:
        audio_shifted = torch.cat([audio[:, :, shift:, :], torch.zeros_like(audio[:, :, :shift, :])], dim=2)
    elif shift < 0:
        shift = abs(shift)
        audio_shifted = torch.cat([torch.zeros_like(audio[:, :, :shift, :]), audio[:, :, :-shift, :]], dim=2)
    else:
        audio_shifted = audio  # No shift
    return audio_shifted

def add_noise(audio, noise_level=0.005):
    """Add Gaussian noise to audio"""
    noise = torch.randn_like(audio) * noise_level
    return audio + noise

def frequency_mask(audio, num_masks=1, max_width=5):
    """Apply frequency masking"""
    freq_dim = audio.shape[3]  # Frequency dimension
    if freq_dim < 2:  
        return audio  # Skip if frequency dimension is too small
    
    for _ in range(num_masks):
        max_width = min(max_width, freq_dim)  # Ensure valid range
        if max_width < 1:
            continue  # Skip if there's no room to mask
        
        f = random.randint(1, max_width)
        f0 = random.randint(0, freq_dim - f)
        audio[:, :, :, f0:f0 + f] = 0
    return audio

def time_mask(audio, num_masks=1, max_width=10):
    """Apply time masking"""
    time_dim = audio.shape[2]  # Time dimension
    if time_dim < 2:  
        return audio  # Skip if time dimension is too small
    
    for _ in range(num_masks):
        max_width = min(max_width, time_dim)  # Ensure valid range
        if max_width < 1:
            continue  # Skip if there's no room to mask
        
        t = random.randint(1, max_width)
        t0 = random.randint(0, time_dim - t)
        audio[:, :, t0:t0 + t, :] = 0
    return audio

def augment_batch(batch, augment_prob=0.5):
    """Apply multiple augmentations with some probability"""
    if random.random() < augment_prob:
        augmentations = [time_shift, add_noise, frequency_mask, time_mask]
        
        # Apply 1-3 random augmentations
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentations, num_augs)
        
        for aug_func in selected_augs:
            batch = aug_func(batch)
    
    return batch
# Normalization functions
def min_max_normalize(batch):
    """Min-max normalization to [0, 1] range"""
    batch_min = batch.min()
    batch_max = batch.max()
    if batch_max - batch_min > 1e-8:  # Avoid division by very small number
        return (batch - batch_min) / (batch_max - batch_min)
    return batch

def z_score_normalize(batch):
    """Z-score normalization"""
    mean = batch.mean()
    std = batch.std()
    if std > 1e-8:  # Avoid division by very small number
        return (batch - mean) / std
    return batch - mean  # If std is too small, just center the data

def db_normalize(batch, ref_db=-120.0):
    """Normalize dB scale audio data"""
    # Assuming batch contains dB values with ref_db as minimum value
    # Map from [ref_db, 0] to [0, 1]
    normalized = (batch - ref_db) / (-ref_db)
    return torch.clamp(normalized, 0, 1)  # Ensure values stay in [0, 1]

# Improved model with batch normalization and dropout
class ImprovedGuitarTabModel(nn.Module):
    def __init__(self, input_channels=1, input_dim=9, time_steps=96, num_classes=19, dropout_rate=0.3):
        super(ImprovedGuitarTabModel, self).__init__()
        
        # Calculate dimensions after convolutions for the flatten operation
        # Input: [batch, channels, time, features]
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Calculate output size after convolutions and pooling
        conv_time = time_steps // 8  # Divided by 2 three times
        conv_features = input_dim // 8  # Divided by 2 three times
        self.flatten_size = 128 * conv_time * conv_features
        
        # Shared layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        # String-specific heads (one for each guitar string)
        self.string_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate/2),  # Less dropout in the final layers
                nn.Linear(256, num_classes)
            ) for _ in range(6)  # 6 guitar strings
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Convolutional blocks - using LeakyReLU instead of standard ReLU
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Shared fully connected layers
        x = self.dropout1(x)
        # print("Final shape before fc1:", x.shape)

        x = F.leaky_relu(self.bn_fc1(self.fc1(x)), negative_slope=0.1)
        # print("Final shape after fc1:", x.shape)

        x = self.dropout2(x)
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)), negative_slope=0.1)
        
        # Apply each string head
        outputs = []
        for head in self.string_heads:
            outputs.append(head(x))
            
        return outputs

# LabelSmoothingLoss for better generalization
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def train_model(model, train_loader, val_loader, epochs=30, device='cuda', lr=0.001):
    # Initialize optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Use label smoothing loss for better generalization
    criterion = LabelSmoothingLoss(classes=19, smoothing=0.1)
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    string_accuracies = [[] for _ in range(6)]
    best_val_loss = float('inf')
    patience = 10  # for early stopping
    counter = 0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs = inputs.to(device)
            # Ensure input is (Batch, Channels, Time, Frequency)
            inputs = inputs.unsqueeze(1)  # (32, 1, 96, 9)
            # print("Input shape before augmentation:", inputs.shape)
            # Apply data augmentation
            # inputs = augment_batch(inputs)
            # print("Input shape after augmentation:", inputs.shape)
            # Apply normalization
            inputs = db_normalize(inputs)
            
            # Process labels
            target_indices = []
            for label in labels:
                if label.dim() > 1 and label.shape[1] > 1:
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    indices = label.to(device).long()
                target_indices.append(indices)
            
            optimizer.zero_grad()
            
            # Ensure input shape matches model's expectation
            if inputs.dim() == 3:  # [batch, time, features]
                inputs = inputs.unsqueeze(1)  # [batch, channel, time, features]
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = 0
            valid_outputs = 0
            
            for output, target in zip(outputs, target_indices):
                if torch.isnan(output).any():
                    continue
                
                if target.dim() != 1:
                    target = target.view(-1)
                
                try:
                    string_loss = criterion(output, target)
                    if not torch.isnan(string_loss).any() and not torch.isinf(string_loss).any():
                        loss += string_loss
                        valid_outputs += 1
                except Exception as e:
                    print(f"Error in loss calculation: {e}")
            
            # Average the loss if we have valid outputs
            if valid_outputs > 0:
                loss = loss / valid_outputs
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
        
        # Calculate average loss per batch
        avg_train_loss = total_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, accuracies = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        for i, acc in enumerate(accuracies):
            string_accuracies[i].append(acc)
        
        # Step the scheduler
        scheduler.step()
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'accuracies': accuracies
            }, 'best_guitar_tab_model.pt')
            print(f"Model saved with validation loss: {val_loss:.4f}")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, string_accuracies)
    
    # Load the best model for final evaluation
    checkpoint = torch.load('best_guitar_tab_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['epoch'], checkpoint['accuracies']

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    batch_count = 0
    
    # Initialize accuracy and confusion matrix trackers
    correct = [0] * 6
    total = [0] * 6
    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            
            # Apply normalization (same as training, but no augmentation)
            inputs = db_normalize(inputs)
            
            # Process labels
            target_indices = []
            for label in labels:
                if label.dim() > 1 and label.shape[1] > 1:
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    indices = label.to(device).long()
                target_indices.append(indices)
            
            # Ensure input shape matches model's expectation
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss and accuracy
            loss = 0
            valid_outputs = 0
            
            for i, (output, target) in enumerate(zip(outputs, target_indices)):
                if torch.isnan(output).any():
                    continue
                
                if target.dim() != 1:
                    target = target.view(-1)
                
                try:
                    string_loss = criterion(output, target)
                    if not torch.isnan(string_loss).any() and not torch.isinf(string_loss).any():
                        loss += string_loss
                        valid_outputs += 1
                        
                        # Calculate accuracy
                        _, predicted = torch.max(output.data, 1)
                        correct[i] += (predicted == target).sum().item()
                        total[i] += target.size(0)
                        
                        # Store predictions and targets for confusion matrix
                        all_preds[i].extend(predicted.cpu().numpy())
                        all_targets[i].extend(target.cpu().numpy())
                except Exception as e:
                    print(f"Error in validation: {e}")
            
            # Average the loss
            if valid_outputs > 0:
                loss = loss / valid_outputs
                total_loss += loss.item()
                batch_count += 1
    
    # Calculate average validation loss
    avg_loss = total_loss / max(batch_count, 1)
    print(f"Validation Loss: {avg_loss:.4f}")
    
    # Calculate and print accuracy for each string
    accuracies = []
    for i in range(6):
        if total[i] > 0:
            accuracy = 100 * correct[i] / total[i]
            accuracies.append(accuracy)
            print(f"Accuracy for string {i+1}: {accuracy:.2f}%")
        else:
            accuracies.append(0)
            print(f"Accuracy for string {i+1}: N/A (no samples)")
    
    # Generate confusion matrices
    plot_confusion_matrices(all_preds, all_targets)
    
    return avg_loss, accuracies

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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    plt.show()  # Display in Colab
    plt.savefig('/content/confusion_matrices.png')  # Save in Colab's filesystem
    print("Confusion matrices saved as 'confusion_matrices.png'")


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


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = ImprovedGuitarTabModel(input_channels=1, input_dim=9, time_steps=96, num_classes=19)
    model = model.to(device)
    print(model)
    
    # Assuming you already have DataLoader objects for training and validation
    # train_loader, val_loader = get_data_loaders()
    # Load data
    image_dir = r'/content/Guitar-Tablature-Classification/cqt_images'
    annotation_dir = r'/content/Guitar-Tablature-Classification/tablature_segments'
    set_seed()
    train_loader, val_loader, test_loader = my_dataloader.create_dataloaders(image_dir, annotation_dir)
    
    # Replace these with your actual loaders
    # For this example, we'll just print instructions
    print("To use this code with your data:")
    print("1. Replace the placeholder train_loader and val_loader with your actual data loaders")
    print("2. Call the main training function:")
    print("   trained_model, best_epoch, final_accuracies = train_model(model, train_loader, val_loader, epochs=30, device=device)")
    
    # Uncomment this when you have your data loaders ready
    trained_model, best_epoch, final_accuracies = train_model(model, train_loader, val_loader, epochs=30, device=device)
    print(f"Best model found at epoch {best_epoch} with accuracies: {final_accuracies}")

    # Testing
    print("Testing the trained model now")
    model.eval() 
    model.to(device)
    test_model(trained_model, test_loader, device)

if __name__ == "__main__":
    main()

# How to use this code:
# 1. Define or load your DataLoaders for training and validation sets
# 2. Create the model and move it to the appropriate device
# 3. Run the training function

# Example usage (uncomment and modify to use):
"""
# Assuming train_loader and val_loader are already defined

# Initialize the model
model = ImprovedGuitarTabModel(input_channels=1, input_dim=9, time_steps=96, num_classes=19)
model = model.to(device)

# Train the model
trained_model, best_epoch, final_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    device=device,
    lr=0.001
)

# Use the trained model for inference
model.eval()
with torch.no_grad():
    # Process a single example
    inputs = sample_input.to(device)
    inputs = inputs.unsqueeze(1)  # Add channel dimension
    outputs = model(inputs)
    
    # Process outputs
    predictions = []
    for output in outputs:
        _, pred = torch.max(output, 1)
        predictions.append(pred)
"""