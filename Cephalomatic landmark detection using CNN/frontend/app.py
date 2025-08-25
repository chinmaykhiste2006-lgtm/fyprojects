import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pandas as pd
import os
from model import CephalometricCNN
from matplotlib.patches import FancyArrowPatch

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Cephalometric Landmark Detector", layout="wide")

# ---------------- TITLE ---------------- #
st.markdown("<h1 style='text-align: center;'>🧠 Cephalometric Landmark Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid gray;'>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
MODEL_PATH = "C:/Users/HP/Desktop/coding/Python/Introduction/CLD ASEP2/Main/resnet_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CephalometricCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- LOAD GROUND TRUTH CSV ---------------- #
CSV_PATH = "C:/Users/HP/Desktop/coding/Python/Introduction/CLD ASEP2/Cephalometric dataset/test2_senior.csv"
ground_truth_df = pd.read_csv(CSV_PATH)

# ---------------- HELPERS ---------------- #
def is_right_facing(image):
    gray = image.convert("L")
    img_np = np.array(gray)
    w = img_np.shape[1]
    return img_np[:, w//2:].mean() > img_np[:, :w//2].mean()

def get_ground_truth(filename):
    matches = ground_truth_df[ground_truth_df['image_path'].str.contains(filename, case=False)]
    if matches.empty:
        return None
    coords = matches.iloc[0, 1:].values.astype(np.float32).reshape(-1, 2)
    x_scale = 512 / 1935
    y_scale = 512 / 2400
    coords[:, 0] *= x_scale
    coords[:, 1] *= y_scale
    return coords

def mean_euclidean_error(pred, gt):
    return np.mean(np.sqrt(np.sum((pred - gt)**2, axis=1)))

def get_quadrant_counts(landmarks):
    center = np.array([256, 256])
    q = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    for x, y in landmarks:
        if x > center[0] and y < center[1]:
            q['Q1'] += 1
        elif x < center[0] and y < center[1]:
            q['Q2'] += 1
        elif x < center[0] and y > center[1]:
            q['Q3'] += 1
        else:
            q['Q4'] += 1
    return q

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("🛠️ Controls")
uploaded_file = st.sidebar.file_uploader("📤 Upload a Cephalometric X-ray", type=["jpg", "png", "bmp"])
show_grid = st.sidebar.checkbox("🧱 Show Grid Overlay", value=True)
marker_size = st.sidebar.slider("🎯 Landmark Size", 10, 100, 40)
enhance = st.sidebar.checkbox("✨ Enhance Image", value=False)
show_ground_truth = st.sidebar.checkbox("🧩 Show Ground Truth Landmarks", value=True)
show_error_arrows = st.sidebar.checkbox("🎯 Show Error Arrows", value=True)
show_index_labels = st.sidebar.checkbox("🔢 Show Landmark Index", value=True)
show_histogram = st.sidebar.checkbox("🌡️ Show Intensity Histogram", value=True)
show_quadrant = st.sidebar.checkbox("📊 Show Quadrant Distribution", value=True)

# ---------------- MAIN ---------------- #
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((512, 512))
    filename = os.path.basename(uploaded_file.name)

    if is_right_facing(image):
        st.sidebar.warning("🔁 Detected right-facing image. Auto-flipping applied.")
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if enhance:
        image = ImageEnhance.Contrast(image).enhance(1.3)
        image = ImageEnhance.Sharpness(image).enhance(2.0)

    image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_landmarks = model(image_tensor).view(19, 2).cpu().numpy()

    gt_landmarks = get_ground_truth(filename) if show_ground_truth else None

    # 🖼️ Image + Prediction Overlay
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🖼️ Input X-ray")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📌 Landmarks Overlay")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)

        for i, (x, y) in enumerate(predicted_landmarks):
            ax.scatter(x, y, c='cyan', s=marker_size, edgecolors='black')
            if show_index_labels:
                ax.text(x + 3, y - 3, str(i+1), color='black', fontsize=8)

        if gt_landmarks is not None:
            ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='red', s=marker_size, marker='x', label="Ground Truth")
            if show_error_arrows:
                for i in range(19):
                    arrow = FancyArrowPatch(posA=(gt_landmarks[i, 0], gt_landmarks[i, 1]),
                                            posB=(predicted_landmarks[i, 0], predicted_landmarks[i, 1]),
                                            arrowstyle="->", color="blue", lw=1, mutation_scale=8)
                    ax.add_patch(arrow)

        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.axis("off")
        ax.legend(loc="upper right")
        st.pyplot(fig)

        # Display MEE
        if gt_landmarks is not None:
            mee = mean_euclidean_error(predicted_landmarks, gt_landmarks)
            st.success(f"✅ Mean Euclidean Error (MEE): {mee:.2f} px")

    # 📍 Predicted Coordinates
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📍 Predicted Landmark Coordinates")
    coords_df = pd.DataFrame(np.round(predicted_landmarks, 2), columns=["X", "Y"])
    st.dataframe(coords_df, use_container_width=True)

    # Download option
    csv_download = coords_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predicted Coordinates", data=csv_download,
                       file_name=f"{filename}_landmarks.csv", mime='text/csv')

    # 🌡️ Intensity Histogram
    if show_histogram:
        gray = image.convert("L")
        pixels = np.array(gray).flatten()
        plt.figure(figsize=(6, 2))
        plt.hist(pixels, bins=50, color="gray", edgecolor="black")
        plt.title("Pixel Intensity Histogram")
        st.pyplot(plt)

    # 📊 Quadrant Plot
    if show_quadrant:
        quadrant_counts = get_quadrant_counts(predicted_landmarks)
        plt.figure(figsize=(4, 3))
        plt.bar(quadrant_counts.keys(), quadrant_counts.values(), color=["blue", "green", "orange", "purple"])
        plt.title("Landmark Distribution by Quadrant")
        plt.ylabel("Count")
        st.pyplot(plt)
