import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps
import random
import torchvision.transforms.functional as TF

# Constants
IMAGE_SIZE = 512
ORIGINAL_SIZE = (2400, 1935)  # (height, width)

class CephalometricDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None, augment=False):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = row["image_path"].strip()
        img_path = os.path.join(self.image_folder, img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file '{img_filename}' not found in {self.image_folder}")

        # Load and resize image to 512×512
        image = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))

        # Load and scale landmarks
        landmarks = row.iloc[1:].values.astype(np.float32)
        x_coords = landmarks[0::2] * (IMAGE_SIZE / ORIGINAL_SIZE[1])
        y_coords = landmarks[1::2] * (IMAGE_SIZE / ORIGINAL_SIZE[0])
        landmarks_scaled = np.stack([x_coords, y_coords], axis=1)

        # Augmentations (only horizontal flip and brightness)
        if self.augment:
            # Horizontal Flip
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                landmarks_scaled[:, 0] = IMAGE_SIZE - landmarks_scaled[:, 0]

            # Brightness Adjustment
            if random.random() < 0.3:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

        # Convert to Tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)  # normalized to [0, 1]

        landmarks_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32)

        return image, landmarks_tensor


# --- Debug/Test Block ---
if __name__ == "__main__":
    # Update the paths accordingly for testing
    dataset_folder = r"C:\Users\HP\Desktop\coding\Python\Introduction\CLD ASEP2\Cephalometric dataset"
    image_folder = os.path.join(dataset_folder, "cepha400")
    csv_path = os.path.join(dataset_folder, "train_senior.csv")

    dataset = CephalometricDataset(csv_path, image_folder, augment=True)
    print(f"✅ Dataset loaded. Total samples: {len(dataset)}")

    # Load a sample
    image_tensor, landmarks_tensor = dataset[0]
    print(f"Image shape: {image_tensor.shape}")  # Should be (3, 512, 512)
    print(f"Landmarks shape: {landmarks_tensor.shape}")  # Should be (19, 2)
    print("First sample landmarks (scaled to 512x512):\n", landmarks_tensor)
