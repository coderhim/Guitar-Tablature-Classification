import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from transformers import ViTImageProcessor


class GuitarTabDataset(Dataset):
    def __init__(self, audio_dir, annotation_dir, img_size=(224, 224)):
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.npy')])
        self.annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.npy')])

        assert len(self.audio_files) == len(self.annotation_files), "Mismatch in audio and annotation file counts."

        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        # Load CQT spectrogram and annotations
        audio = np.load(audio_path).astype(np.float32)  # CQT data in dB scale
        annotation = np.load(annotation_path).astype(np.float32)

        # Normalize the CQT data to [0, 1] range
        audio_normalized = (audio + 120) / 120  # Assuming the minimum is -120 dB
        audio_normalized = np.clip(audio_normalized, 0, 1)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_normalized)
        
        # Add channel dimension if needed
        if len(audio_tensor.shape) == 2:  # (H, W)
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, H, W)
        
        # Resize to the target image size
        audio_tensor = torch.nn.functional.interpolate(
            audio_tensor.unsqueeze(0),  # Add batch dimension for interpolate
            size=self.img_size, 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Convert to 3 channels for ViT (typically expects RGB)
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(3, 1, 1)  # (1, H, W) → (3, H, W)

        # Process annotations (6 strings for guitar)
        heads = [torch.tensor(annotation[i]).long() for i in range(6)]  # Convert to long for classification
        
        return audio_tensor, heads


def create_dataloaders(audio_dir, annotation_dir, batch_size=50, train_ratio=0.8, val_ratio=0.1, img_size=(224, 224)):
    dataset = GuitarTabDataset(audio_dir, annotation_dir, img_size)

    # Split dataset into train, validation, and test
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Use a fixed random seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # DataLoader configuration
    loader_args = {
        'batch_size': batch_size,
        'num_workers': min(4, os.cpu_count() or 1),
        'pin_memory': torch.cuda.is_available(),
    }
    
    # Add prefetch_factor only if num_workers > 0
    if loader_args['num_workers'] > 0:
        loader_args['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader