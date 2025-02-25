import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import GuitarTabNet
import my_dataloader

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

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            
            # For debugging
            # for i, label in enumerate(labels):
            #     print(f"Label {i} shape before processing: {label.shape}")
            
            # Convert one-hot encoded labels to class indices
            # This assumes your labels are one-hot encoded with shape [batch_size, num_classes]
            target_indices = []
            for label in labels:
                # If it's one-hot encoded, convert to indices
                if label.dim() > 1 and label.shape[1] > 1:
                    # Get the index of the maximum value along dimension 1
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    # If already indices, just ensure it's 1D
                    indices = label.view(-1).to(device).long()
                target_indices.append(indices)
            
            optimizer.zero_grad()

            # Ensure input shape matches model's expectation
            inputs = inputs.unsqueeze(1)  # Adding channel dimension if missing

            outputs = model(inputs)

            # Calculate loss for each output-label pair and sum them
            loss = 0
            for output, target in zip(outputs, target_indices):
                loss += criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

        # Evaluate on validation set
        validate_model(model, val_loader, criterion)

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = [0] * 6
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            
            # Convert one-hot encoded labels to class indices
            target_indices = []
            for label in labels:
                # If it's one-hot encoded, convert to indices
                if label.dim() > 1 and label.shape[1] > 1:
                    indices = torch.argmax(label, dim=1).to(device)
                else:
                    indices = label.view(-1).to(device).long()
                target_indices.append(indices)

            inputs = inputs.unsqueeze(1)  # Ensure input shape is consistent

            outputs = model(inputs)

            # Calculate validation loss
            batch_loss = 0
            for output, target in zip(outputs, target_indices):
                batch_loss += criterion(output, target)
            val_loss += batch_loss.item()

            # Calculate accuracy for each output
            for i in range(6):
                pred = torch.argmax(outputs[i], dim=1)
                correct[i] += (pred == target_indices[i]).sum().item()

            total += target_indices[0].size(0)

    print(f"Validation Loss: {val_loss:.4f}")
    for i, acc in enumerate(correct):
        print(f"Accuracy for string {i + 1}: {acc / total * 100:.2f}%")

if __name__ == '__main__':
    # Load data
    audio_dir = r'D:\Code playground\seminar_audioTab_\cqt_audio'
    annotation_dir = r'D:\Code playground\seminar_audioTab_\tablature_segments'

    train_loader, val_loader, test_loader = my_dataloader.create_dataloaders(audio_dir, annotation_dir)

    # Inspect a sample batch
    for audio, heads in train_loader:
        print("Audio shape:", audio.shape)
        print("Head shapes:", [h.shape for h in heads])
        break

    # Start training
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs)