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
from my_dataloader import create_dataloaders
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
from ViT_model import DinoGuitarTabModel
from transformers import AutoModel, get_cosine_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import torch
from matplotlib.ticker import MaxNLocator
import pandas as pd
from tqdm import tqdm
import time
# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed()

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

def check_tensor(tensor, name="Input"):
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} min: {tensor.min().item()}, max: {tensor.max().item()}")
    print(f"{name} mean: {tensor.mean().item()}, std: {tensor.std().item()}")
    print(f"{name} unique values: {torch.unique(tensor).shape[0]}")



def train_model(model, train_loader, val_loader, epochs=30, device='cuda', lr=0.0005):
    # Initialize optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    # Use label smoothing loss for better generalization
    criterion = LabelSmoothingLoss(classes=19, smoothing=0.1)
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    string_accuracies = [[] for _ in range(6)]
    string_f1_scores = [[] for _ in range(6)]
    string_precisions = [[] for _ in range(6)]
    string_recalls = [[] for _ in range(6)]
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
        val_metrics = validate_model(model, val_loader, criterion, device)
        val_loss, accuracies, f1_scores, precisions, recalls = val_metrics
        val_losses.append(val_loss)
        
        for i in range(6):
            string_accuracies[i].append(accuracies[i])
            string_f1_scores[i].append(f1_scores[i])
            string_precisions[i].append(precisions[i])
            string_recalls[i].append(recalls[i])
        
        # Step the scheduler
        scheduler.step()
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Print F1 scores for each string
        print("F1 Scores:")
        for i, f1 in enumerate(f1_scores):
            print(f"  String {i+1}: {f1:.4f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'accuracies': accuracies,
                'f1_scores': f1_scores,
                'precisions': precisions,
                'recalls': recalls
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
    plot_training_metrics(train_losses, val_losses, string_accuracies, 
                         string_f1_scores, string_precisions, string_recalls)
    
    # Load the best model for final evaluation
    checkpoint = torch.load('best_vit_guitar_tab_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['epoch'], checkpoint['accuracies'], checkpoint['f1_scores']

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    batch_count = 0
    
    # Initialize accuracy and metrics trackers
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
                        
                        # Store predictions and targets for confusion matrix and other metrics
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
    
    # Calculate metrics for each string
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for i in range(6):
        if total[i] > 0:
            # Calculate accuracy
            accuracy = 100 * correct[i] / total[i]
            accuracies.append(accuracy)
            
            # Calculate F1, precision, and recall (with handling for potential warnings)
            if len(all_targets[i]) > 0 and len(np.unique(all_targets[i])) > 1:
                # For macro averaging (treats all classes equally regardless of imbalance)
                f1 = f1_score(all_targets[i], all_preds[i], average='macro')
                precision = precision_score(all_targets[i], all_preds[i], average='macro')
                recall = recall_score(all_targets[i], all_preds[i], average='macro')
            else:
                f1, precision, recall = 0, 0, 0
                
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            
            print(f"String {i+1} - Accuracy: {accuracy:.2f}%, F1: {f1:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        else:
            accuracies.append(0)
            f1_scores.append(0)
            precisions.append(0)
            recalls.append(0)
            print(f"String {i+1}: N/A (no samples)")
    
    # Generate confusion matrices and classification reports
    plot_confusion_matrices(all_preds, all_targets)
    print_classification_reports(all_preds, all_targets)
    
    return avg_loss, accuracies, f1_scores, precisions, recalls

def print_classification_reports(all_preds, all_targets):
    """Print detailed classification reports for each string."""
    print("\n=== Classification Reports ===")
    
    for i in range(6):
        if len(all_targets[i]) > 0 and len(all_preds[i]) > 0:
            print(f"\nString {i+1} Classification Report:")
            try:
                report = classification_report(all_targets[i], all_preds[i])
                print(report)
            except Exception as e:
                print(f"Could not generate report: {e}")
        else:
            print(f"\nString {i+1}: No data available")

def plot_training_metrics(train_losses, val_losses, string_accuracies, 
                         string_f1_scores, string_precisions, string_recalls):
    """
    Plot training metrics including loss, accuracy, F1 score, precision, and recall.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(20, 15))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    for i in range(6):
        plt.plot(epochs, string_accuracies[i], label=f'String {i+1}')
    plt.title('Validation Accuracy per String')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 scores
    plt.subplot(2, 2, 3)
    for i in range(6):
        plt.plot(epochs, string_f1_scores[i], label=f'String {i+1}')
    plt.title('F1 Scores per String')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot precision and recall
    plt.subplot(2, 2, 4)
    
    # Create two separate line styles for precision and recall
    for i in range(6):
        plt.plot(epochs, string_precisions[i], linestyle='-', 
                marker='o', markersize=3, label=f'Precision String {i+1}')
        plt.plot(epochs, string_recalls[i], linestyle='--', 
                marker='x', markersize=3, label=f'Recall String {i+1}')
    
    plt.title('Precision and Recall per String')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend(fontsize='small')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()
    
    # Additional plots for detailed per-string metrics
    plot_detailed_string_metrics(epochs, string_accuracies, string_f1_scores, 
                               string_precisions, string_recalls)

def plot_detailed_string_metrics(epochs, string_accuracies, string_f1_scores, 
                               string_precisions, string_recalls):
    """Create detailed per-string metric plots."""
    for i in range(6):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, string_accuracies[i], 'b-')
        plt.title(f'String {i+1} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, string_f1_scores[i], 'g-')
        plt.title(f'String {i+1} F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs, string_precisions[i], 'r-')
        plt.title(f'String {i+1} Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(epochs, string_recalls[i], 'm-')
        plt.title(f'String {i+1} Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'string_{i+1}_metrics.png', dpi=300)
        plt.close()

def plot_confusion_matrices(all_preds, all_targets):
    """Plot confusion matrices for each string with improved visualization."""
    for i in range(6):
        if len(all_targets[i]) > 0 and len(all_preds[i]) > 0:
            # Get all unique classes
            classes = sorted(list(set(all_targets[i] + all_preds[i])))
            
            if len(classes) > 1:  # Only create matrix if we have multiple classes
                cm = confusion_matrix(all_targets[i], all_preds[i], labels=classes)
                
                # Create a normalized version for easier interpretation
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
                
                # Plot the confusion matrix
                plt.figure(figsize=(10, 8))
                
                # Plot the raw counts
                plt.subplot(1, 2, 1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=classes, yticklabels=classes)
                plt.title(f'String {i+1} Confusion Matrix (Counts)')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                
                # Plot the normalized percentages
                plt.subplot(1, 2, 2)
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                          xticklabels=classes, yticklabels=classes)
                plt.title(f'String {i+1} Confusion Matrix (Normalized)')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                
                plt.tight_layout()
                plt.savefig(f'string_{i+1}_confusion_matrix.png', dpi=300)
                plt.close()

# Additional function to generate per-class metrics for better understanding
def plot_per_class_metrics(all_preds, all_targets):
    """Plot per-class metrics for each string."""
    for i in range(6):
        if len(all_targets[i]) == 0 or len(np.unique(all_targets[i])) <= 1:
            continue
            
        # Get all unique classes
        classes = sorted(list(set(all_targets[i])))
        
        # Calculate per-class precision, recall, and F1
        precisions = []
        recalls = []
        f1s = []
        
        for cls in classes:
            true_positives = sum((np.array(all_targets[i]) == cls) & (np.array(all_preds[i]) == cls))
            false_positives = sum((np.array(all_targets[i]) != cls) & (np.array(all_preds[i]) == cls))
            false_negatives = sum((np.array(all_targets[i]) == cls) & (np.array(all_preds[i]) != cls))
            
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-5)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        # Create a DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            'Class': classes,
            'Precision': precisions,
            'Recall': recalls,
            'F1': f1s
        })
        
        # Plot the metrics
        plt.figure(figsize=(12, 6))
        
        # Sort by F1 score for better visualization
        metrics_df = metrics_df.sort_values('F1', ascending=False)
        
        # Create a bar chart
        x = np.arange(len(metrics_df))
        width = 0.25
        
        plt.bar(x - width, metrics_df['Precision'], width, label='Precision')
        plt.bar(x, metrics_df['Recall'], width, label='Recall')
        plt.bar(x + width, metrics_df['F1'], width, label='F1')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title(f'String {i+1} Per-Class Metrics')
        plt.xticks(x, metrics_df['Class'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'string_{i+1}_per_class_metrics.png', dpi=300)
        plt.close()

def test_model(model, test_loader, device):
    model.eval()
    correct = [0] * 6
    total = [0] * 6
    
    # For storing predictions and targets for metric calculation
    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]

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
                
                # Store predictions and targets for metrics
                all_preds[i].extend(predicted.cpu().numpy())
                all_targets[i].extend(target.cpu().numpy())

    # Calculate and report all metrics
    print("\n=== Test Results ===")
    print("String | Accuracy | F1 Score | Precision | Recall")
    print("------|----------|----------|-----------|-------")
    
    # For storing metrics to return
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for i in range(6):
        if total[i] > 0:
            # Calculate accuracy
            accuracy = 100 * correct[i] / total[i]
            
            # Calculate F1, precision, and recall
            if len(all_targets[i]) > 0 and len(np.unique(all_targets[i])) > 1:
                f1 = f1_score(all_targets[i], all_preds[i], average='macro')
                precision = precision_score(all_targets[i], all_preds[i], average='macro')
                recall = recall_score(all_targets[i], all_preds[i], average='macro')
            else:
                f1, precision, recall = 0, 0, 0
                
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            
            print(f"  {i+1}   | {accuracy:6.2f}% | {f1:.4f}   | {precision:.4f}    | {recall:.4f}")
        else:
            accuracies.append(0)
            f1_scores.append(0)
            precisions.append(0)
            recalls.append(0)
            print(f"  {i+1}   | N/A       | N/A      | N/A        | N/A")
    
    # Generate confusion matrices and classification reports
    plot_confusion_matrices(all_preds, all_targets)
    print_classification_reports(all_preds, all_targets)
    
    # Plot per-class metrics for deeper analysis
    plot_per_class_metrics(all_preds, all_targets)
    
    # Calculate overall metrics (averaged across all strings)
    overall_accuracy = sum([acc * tot for acc, tot in zip(accuracies, total)]) / sum(total) if sum(total) > 0 else 0
    overall_f1 = sum(f1_scores) / sum(1 for f1 in f1_scores if f1 > 0) if any(f1 > 0 for f1 in f1_scores) else 0
    overall_precision = sum(precisions) / sum(1 for p in precisions if p > 0) if any(p > 0 for p in precisions) else 0
    overall_recall = sum(recalls) / sum(1 for r in recalls if r > 0) if any(r > 0 for r in recalls) else 0
    
    print("\n=== Overall Model Performance ===")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    
    # Generate summary visualization
    plot_test_results_summary(accuracies, f1_scores, precisions, recalls)
    
    return accuracies, f1_scores, precisions, recalls

def plot_test_results_summary(accuracies, f1_scores, precisions, recalls):
    """Create a summary visualization of test results."""
    plt.figure(figsize=(12, 8))
    
    strings = [f"String {i+1}" for i in range(6)]
    x = np.arange(len(strings))
    width = 0.2
    
    # Convert accuracy to same scale as other metrics (0-1)
    normalized_accuracies = [acc/100 for acc in accuracies]
    
    plt.bar(x - width*1.5, normalized_accuracies, width, label='Accuracy', color='blue')
    plt.bar(x - width/2, f1_scores, width, label='F1 Score', color='green')
    plt.bar(x + width/2, precisions, width, label='Precision', color='red')
    plt.bar(x + width*1.5, recalls, width, label='Recall', color='purple')
    
    plt.xlabel('Guitar String')
    plt.ylabel('Score')
    plt.title('Test Metrics by Guitar String')
    plt.xticks(x, strings)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(normalized_accuracies):
        plt.text(i - width*1.5, v + 0.02, f"{accuracies[i]:.1f}%", ha='center', fontsize=9)
    
    for i, v in enumerate(f1_scores):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
        
    for i, v in enumerate(precisions):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
        
    for i, v in enumerate(recalls):
        plt.text(i + width*1.5, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('test_results_summary.png', dpi=300)
    plt.show()

# Create a function to generate ROC curves for each string (if applicable)
def plot_roc_curves(model, test_loader, device):
    """Plot ROC curves for each string if possible."""
    from sklearn.metrics import roc_curve, auc
    
    model.eval()
    
    # For storing probabilities and true labels
    all_probs = [[] for _ in range(6)]
    all_labels = [[] for _ in range(6)]
    
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
            
            # Store softmax probabilities and true labels
            for i, (output, target) in enumerate(zip(outputs, target_indices)):
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
                all_probs[i].extend(probs)
                all_labels[i].extend(target.cpu().numpy())
    
    # Plot ROC curves for each string
    plt.figure(figsize=(15, 10))
    
    for i in range(6):
        if len(all_labels[i]) > 0 and len(np.unique(all_labels[i])) > 1:
            plt.subplot(2, 3, i+1)
            
            # Get all unique classes
            classes = sorted(list(set(all_labels[i])))
            
            # One-vs-Rest ROC curve for each class
            for c in classes:
                # Convert to binary classification problem
                y_true = np.array(all_labels[i]) == c
                y_score = np.array(all_probs[i])[:, c]
                
                # Calculate ROC
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'Class {c} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'String {i+1} ROC Curve')
            plt.legend(loc="lower right", fontsize='small')
        else:
            plt.subplot(2, 3, i+1)
            plt.text(0.5, 0.5, 'Insufficient data', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title(f'String {i+1}')
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed()
    # Create model and move to device
    # model = ViTGuitarTabModel(num_classes=19, pretrained_model="google/vit-base-patch16-224")
    # Instantiate the model using facebook/dino-vit-small-patch8
    # model = DinoGuitarTabModel(num_classes=19, pretrained_model="facebook/dino-vits8")
    model = DinoGuitarTabModel()

    model = model.to(device)
    print(model)
    
    # Load data
    image_dir = r'/content/Guitar-Tablature-Classification/cqt_images'
    annotation_dir = r'/content/Guitar-Tablature-Classification/tablature_segments'

    train_loader, val_loader, test_loader = create_dataloaders(image_dir, annotation_dir)
    
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