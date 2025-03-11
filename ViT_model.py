import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTImageProcessor

class ViTGuitarTabModel(nn.Module):
    def __init__(self, num_classes=19, dropout_rate=0.3, pretrained_model="facebook/dino-vits8"):
        super(ViTGuitarTabModel, self).__init__()
        
        # Load pre-trained ViT model & processor
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model)
        self.vit = ViTModel.from_pretrained(pretrained_model)
        
        # Adjust hidden size for dino-vit-small (384 for small models)
        vit_output_dim = self.vit.config.hidden_size  # 384 for dino-vit-small

        # Shared layers after ViT backbone
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(vit_output_dim, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        # String-specific heads (one for each guitar string)
        self.string_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate / 2),  # Less dropout in final layers
                nn.Linear(256, num_classes)
            ) for _ in range(6)  # 6 guitar strings
        ])

        # Initialize custom layers
        self._init_custom_weights()

    def _init_custom_weights(self):
        # Initialize weights for the fully connected layers
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        for m in [self.bn_fc1, self.bn_fc2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        # Initialize string head layers
        for head in self.string_heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def reshape_for_vit(self, x):
        """Ensure input is resized and formatted for ViT"""
        # If input is [batch, time, freq], add channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, time, freq]
        
        # Resize to 128x128 (for dino-vit-small-patch8)
        if x.shape[2] != 128 or x.shape[3] != 128:
            x = F.interpolate(x, size=(128, 128), mode='bicubic', align_corners=False)
        
        # ViT can work with 1 channel inputs—no need to repeat
        return x

    def forward(self, x):
        # Preprocess and reshape for ViT
        x = self.reshape_for_vit(x)

        # Process with ViT
        inputs = self.processor(images=x, return_tensors="pt")
        outputs = self.vit(**inputs)

        # Extract [CLS] token representation
        x = outputs.last_hidden_state[:, 0]  # Shape: [batch_size, hidden_size]

        # Pass through shared fully connected layers
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)), negative_slope=0.1)
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)), negative_slope=0.1)

        # Apply string-specific heads
        outputs = [head(x) for head in self.string_heads]

        return outputs
