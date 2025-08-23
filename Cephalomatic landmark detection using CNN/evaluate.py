import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import CephalometricDataset
from model import CephalometricCNN

# ------------------ Constants ------------------ #
IMAGE_SIZE = 512
ORIGINAL_SIZE = (2400, 1935)  # For future use if resizing back is needed
MODEL_PATH = r"C:\Users\HP\Desktop\coding\Python\Introduction\CLD ASEP2\best_model.pth"
PIXEL_TO_MM = 0.265  # Approximate pixel-to-mm conversion

# ------------------ Paths ------------------ #
dataset_folder = r"C:\Users\HP\Desktop\coding\Python\Introduction\CLD ASEP2\Cephalometric dataset"
image_folder = os.path.join(dataset_folder, "cepha400")
test_csv_path = os.path.join(dataset_folder, "test2_senior.csv")

# ------------------ Load Test Dataset ------------------ #
test_dataset = CephalometricDataset(test_csv_path, image_folder)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------ Load Trained Model ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CephalometricCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ------------------ MEE Calculation ------------------ #
def mean_euclidean_error(predictions, targets):
    return torch.mean(torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1)))

# ------------------ Evaluate & Visualize ------------------ #
total_error = 0.0
num_samples = 0

output_dir = r"C:\Users\HP\Desktop\coding\Python\Introduction\CLD ASEP2\output_images"
os.makedirs(output_dir, exist_ok=True)

for idx, (image, true_landmarks) in enumerate(test_loader):
    image, true_landmarks = image.to(device), true_landmarks.to(device)

    with torch.no_grad():
        predicted_landmarks = model(image)

    predicted_landmarks = predicted_landmarks.view(19, 2)
    error = mean_euclidean_error(predicted_landmarks, true_landmarks.squeeze())
    total_error += error.item()
    num_samples += 1

    # Print per-sample MEE
    print(f"Sample {idx + 1}: MEE = {error.item():.2f} px")

    # Visualize result
    image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
    true_lm_np = true_landmarks.cpu().numpy().squeeze()
    pred_lm_np = predicted_landmarks.cpu().numpy()

    plt.imshow(image_np, cmap='gray')
    plt.scatter(true_lm_np[:, 0], true_lm_np[:, 1], c='green', label="Ground Truth", s=20)
    plt.scatter(pred_lm_np[:, 0], pred_lm_np[:, 1], c='red', label="Predicted", s=20)
    plt.title(f"Test Sample {idx + 1}")
    plt.legend()
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"output_image_{idx + 1}.png"))
    plt.close()

# ------------------ Final Report ------------------ #
if num_samples == 0:
    print("⚠️ No samples found in test loader. Check your CSV paths or data format.")
else:
    avg_error = total_error / num_samples
    avg_error_mm = avg_error * PIXEL_TO_MM
    print(f"\n✅ Mean Euclidean Error (MEE) on Test Set: {avg_error:.2f} px / {avg_error_mm:.2f} mm")

    # Save to file
    with open("mee_score.txt", "w") as f:
        f.write(f"{avg_error:.2f} px / {avg_error_mm:.2f} mm")
