import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data_pipeline import get_dataloader

def train_model(net, epochs=2):
    print("Training model with DataLoader...")
    loader = get_dataloader()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        correct, total = 0, 0
        running_loss = 0.0

        for imgs, labels in tqdm(loader):
            outputs = net(imgs)  # imgs already in [B, 3, H, W]
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()

            # Track accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        print(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}")

    # ✅ Save model after training
    torch.save(net.state_dict(), "models/resnet18_crack.pth")
    print("💾 Model saved to models/resnet18_crack.pth")
