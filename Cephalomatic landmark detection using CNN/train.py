import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CephalometricDataset
from model import CephalometricCNN
from utils import mean_euclidean_error
import matplotlib.pyplot as plt

print("✅ Imports Successful!")

# ------------------ Hyperparameters ------------------ #
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ------------------ Paths ------------------ #
base_path = os.path.join(os.getcwd(), "Cephalometric dataset")
image_folder = os.path.join(base_path, "cepha400")
train_csv_path = os.path.join(base_path, "train_senior.csv")
val_csv_path = os.path.join(base_path, "test1_senior.csv")

print("Train CSV:", train_csv_path)
print("Val CSV:", val_csv_path)

# ------------------ Data Loaders ------------------ #
train_dataset = CephalometricDataset(train_csv_path, image_folder, augment=True)
val_dataset = CephalometricDataset(val_csv_path, image_folder, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ------------------ Model ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CephalometricCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------ Training ------------------ #
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, landmarks in train_loader:
        images = images.to(device)
        landmarks = landmarks.to(device)

        optimizer.zero_grad()
        preds = model(images)  # Output shape: (B, 38)
        loss = criterion(preds, landmarks.view(landmarks.size(0), -1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ------------------ Validation ------------------ #
    model.eval()
    val_loss = 0.0
    total_mee_px = 0.0
    total_mee_mm = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, landmarks in val_loader:
            images = images.to(device)
            landmarks = landmarks.to(device)

            preds = model(images)              # (B, 38)
            preds = preds.view(-1, 19, 2)      # (B, 19, 2)
            true = landmarks.view(-1, 19, 2)   # (B, 19, 2)

            loss = criterion(preds.view(preds.size(0), -1), true.view(true.size(0), -1))
            val_loss += loss.item()

            mee_px, mee_mm = mean_euclidean_error(preds, true)
            total_mee_px += mee_px * preds.size(0)  # Multiply by batch size
            total_mee_mm += mee_mm * preds.size(0)
            total_samples += preds.size(0)

    avg_val_loss = val_loss / len(val_loader)
    avg_mee_px = total_mee_px / total_samples
    avg_mee_mm = total_mee_mm / total_samples

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MEE: {avg_mee_px:.2f} px / {avg_mee_mm:.2f} mm")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")

print("🎉 Training Completed!")
