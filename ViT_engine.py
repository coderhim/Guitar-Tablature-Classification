import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import random
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor
from ViT_dataloader import create_dataloaders

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Data augmentation: Random shifts and noise
def augment_batch(batch, augment_prob=0.5):
    if random.random() < augment_prob:
        noise = torch.randn_like(batch) * 0.005
        batch += noise
    return batch

# ViT Model for Guitar Tab Classification
class ViTGuitarTabModel(nn.Module):
    def __init__(self, num_classes=19, pretrained_model="google/vit-base-patch16-224"):
        super(ViTGuitarTabModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(pretrained_model, num_labels=num_classes * 6)
        self.num_strings = 6
        self.num_classes = num_classes

    def forward(self, x):
        outputs = self.vit(x).logits
        outputs = outputs.view(-1, self.num_strings, self.num_classes)
        return [outputs[:, i, :] for i in range(self.num_strings)]

# Label smoothing loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def train_model(model, train_loader, val_loader, epochs=30, device='cuda', lr=0.0001):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = LabelSmoothingLoss(classes=19, smoothing=0.1)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = batch
            inputs = augment_batch(inputs.to(device))
            targets = [target.to(device) for target in targets]

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = sum(criterion(output, target) for output, target in zip(outputs, targets))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}")

        validate_model(model, val_loader, criterion, device)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = [target.to(device) for target in targets]

            outputs = model(inputs)
            loss = sum(criterion(output, target) for output, target in zip(outputs, targets))

            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Paths to your dataset folders (update these paths)
    audio_dir = r'/content/Guitar-Tablature-Classification/cqt_audio'
    annotation_dir = r'/content/Guitar-Tablature-Classification/tablature_segments'

    # Create the dataloaders
    batch_size = 32
    train_loader, val_loader, test_loader = create_dataloaders(audio_dir, annotation_dir, batch_size=batch_size)

    # train_loader, val_loader = my_dataloader.load_data(batch_size=32)

    model = ViTGuitarTabModel()
    model.to(device)
    train_model(model, train_loader, val_loader)
    # Save the model weights
    torch.save(model.state_dict(), "guitar_vit_model.pth")
    print("Model weights saved successfully!")
    model.load_state_dict(torch.load("guitar_vit_model.pth"))
    model.eval()

    # Evaluate on the test set
    with torch.no_grad():
        for audio, heads in test_loader:
            audio = audio.to(device)
            outputs = torch.sigmoid(model(audio))  # Convert logits to probabilities
            print(outputs)
            break  # Check the first batch
