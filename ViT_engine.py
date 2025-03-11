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
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
from ViT_model import ViTGuitarTabModel
# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Keep the data augmentation and normalization functions from your original code
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

# # New model using Vision Transformer
# class ViTGuitarTabModel(nn.Module):
#     def __init__(self, num_classes=19, dropout_rate=0.3, pretrained_model="google/vit-base-patch16-224"):
#         super(ViTGuitarTabModel, self).__init__()
        
#         # Load pre-trained ViT model
#         self.vit = ViTModel.from_pretrained(pretrained_model)
        
#         # Freeze the base ViT model if needed (optional)
#         # Uncomment the following line to freeze the base model
#         # for param in self.vit.parameters():
#         #    param.requires_grad = False
        
#         # Get the dimensionality of the ViT's output
#         vit_output_dim = self.vit.config.hidden_size  # typically 768 for base model
        
#         # Shared layers after ViT backbone
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.fc1 = nn.Linear(vit_output_dim, 512)
#         self.bn_fc1 = nn.BatchNorm1d(512)
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn_fc2 = nn.BatchNorm1d(256)
        
#         # String-specific heads (one for each guitar string)
#         self.string_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Dropout(dropout_rate/2),  # Less dropout in the final layers
#                 nn.Linear(256, num_classes)
#             ) for _ in range(6)  # 6 guitar strings
#         ])
        
#         # Initialize weights for our custom layers
#         self._init_custom_weights()
        
#     def _init_custom_weights(self):
#         # Initialize weights for the fully connected layers
#         for m in [self.fc1, self.fc2]:
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
                
#         for m in [self.bn_fc1, self.bn_fc2]:
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
            
#         # Initialize string head layers
#         for head in self.string_heads:
#             for layer in head:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
#                     if layer.bias is not None:
#                         nn.init.constant_(layer.bias, 0)
    
    # def reshape_for_vit(self, x):
    #     """Reshape input to match ViT's expected input format"""
    #     batch_size = x.shape[0]
        
    #     # Check input shape
    #     if x.dim() == 3:  # [batch, time, freq]
    #         x = x.unsqueeze(1)  # Add channel dim: [batch, channel, time, freq]
        
    #     # ViT expects input shape [batch_size, channels, height, width]
    #     # where height and width are typically 224x224 for standard ViT models
        
    #     # Resize to 224x224 if needed
    #     if x.shape[2] != 224 or x.shape[3] != 224:
    #         # Option 1: Use interpolate to resize
    #         x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            
    #     # If we have only 1 channel, repeat it to create 3 channels (RGB)
    #     if x.shape[1] == 1:
    #         x = x.repeat(1, 3, 1, 1)
            
    #     return x
    
    # def forward(self, x):
    #     # Reshape for ViT
    #     x = self.reshape_for_vit(x)
        
    #     # Forward pass through ViT
    #     outputs = self.vit(pixel_values=x)
        
    #     # Get the [CLS] token output which represents the entire sequence
    #     x = outputs.last_hidden_state[:, 0]  # Shape: [batch_size, hidden_size]
        
    #     # Shared fully connected layers
    #     x = self.dropout1(x)
    #     x = F.leaky_relu(self.bn_fc1(self.fc1(x)), negative_slope=0.1)
    #     x = self.dropout2(x)
    #     x = F.leaky_relu(self.bn_fc2(self.fc2(x)), negative_slope=0.1)
        
    #     # Apply each string head
    #     outputs = []
    #     for head in self.string_heads:
    #         outputs.append(head(x))
            
    #     return outputs

# LabelSmoothingLoss (keep this from your original code)
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

def train_model(model, train_loader, val_loader, epochs=30, device='cuda', lr=0.0005):
    # Initialize optimizer with weight decay for regularization
    # Use a lower learning rate for the pre-trained model
    optimizer = torch.optim.AdamW([
        {'params': model.vit.parameters(), 'lr': lr / 10},  # Lower LR for pre-trained parameters
        {'params': model.fc1.parameters()},
        {'params': model.fc2.parameters()},
        {'params': model.bn_fc1.parameters()},
        {'params': model.bn_fc2.parameters()},
        {'params': model.string_heads.parameters()}
    ], lr=lr, weight_decay=1e-4)
    
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
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)  # (batch, 1, time, freq)
                
            # Apply data augmentation
            # inputs = augment_batch(inputs)
            print(inputs)
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
              f"LR: {optimizer.param_groups[1]['lr']:.6f}")
        
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
            }, 'best_vit_guitar_tab_model.pt')
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
    checkpoint = torch.load('best_vit_guitar_tab_model.pt')
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
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
                
            inputs = db_normalize(inputs)
            
            # Process labels
            target_indices = []
            for label in labels:
                if label.dim() > 1 and label.shape[1] > 1:
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    indices = label.to(device).long()
                target_indices.append(indices)
            
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

# Keep these functions from your original code
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
    plt.show()
    plt.savefig('vit_training_metrics.png')
    plt.close()

def plot_confusion_matrices(all_preds, all_targets):
    """Plot confusion matrices for each string."""
    plt.figure(figsize=(20, 15))
    
    for i in range(6):
        if len(all_preds[i]) > 0:
            cm = confusion_matrix(all_targets[i], all_preds[i])
            plt.subplot(2, 3, i+1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - String {i+1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('vit_confusion_matrices.png')
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and move to device
    # model = ViTGuitarTabModel(num_classes=19, pretrained_model="google/vit-base-patch16-224")
    # Instantiate the model using facebook/dino-vit-small-patch8
    model = ViTGuitarTabModel(num_classes=19, pretrained_model="facebook/dino-vits8")

    model = model.to(device)
    print(model)
    
    # Load data
    audio_dir = r'/content/Guitar-Tablature-Classification/cqt_audio'
    annotation_dir = r'/content/Guitar-Tablature-Classification/tablature_segments'

    train_loader, val_loader, test_loader = my_dataloader.create_dataloaders(audio_dir, annotation_dir)
    
    # Train the model
    trained_model, best_epoch, final_accuracies = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=30, 
        device=device, 
        lr=0.0005  # Lower learning rate for fine-tuning
    )
    
    print(f"Best model found at epoch {best_epoch} with accuracies: {final_accuracies}")

    # Testing
    print("Testing the trained model now")
    trained_model.eval() 
    trained_model.to(device)
    test_model(trained_model, test_loader, device)

if __name__ == "__main__":
    main()