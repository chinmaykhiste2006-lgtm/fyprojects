# 🧠 Cephalometric Landmark Detection  

<p align="center">
  <em>Deep Learning for Automated Orthodontic Landmark Detection</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
  <img src="https://img.shields.io/badge/Contributions-Welcome-orange" />
</p>

---

## 📌 Introduction  
Cephalometric analysis is an essential step in orthodontic diagnosis and treatment planning. Traditionally, clinicians manually annotate anatomical landmarks on cephalograms — a process that is time-consuming and prone to variability.  

This project implements an **AI-powered Cephalometric Landmark Detection system** using a **ResNet18-based CNN** to automatically localize 19 key landmarks on cephalometric radiographs. Our approach reduces manual effort, improves consistency, and demonstrates how deep learning can assist in medical imaging tasks.  

---

## ✨ Features  
- 🖼️ **Automatic landmark detection** on cephalometric X-rays.  
- ⚡ **ResNet18 backbone** for robust feature extraction.  
- 🔄 **Data preprocessing & augmentation** (resizing, flipping, brightness).  
- 📊 **Scalable pipeline** for training and evaluation.  
- 🧪 Outputs **38 coordinates** (x, y for each of 19 landmarks).  

---

## 📂 Dataset  
- Images: Cephalometric radiographs (original size: **2400 × 1935**).  
- Labels: CSV files containing 19 anatomical landmarks per image.  
- Preprocessing: Images resized to **512 × 512**, landmarks scaled accordingly.  

⚠️ Dataset is private/academic and not included here.  

---

## 🏗️ Model Architecture  
- Backbone: **ResNet18 (pretrained on ImageNet)**, truncated before final layers.  
- Global Average Pooling: Reduces feature maps to (512, 1, 1).  
- Regression Head:  
  - Linear(512 → 256) + ReLU + Dropout(0.3)  
  - Linear(256 → 38) → Outputs (x, y) pairs for 19 landmarks.  

---

## ⚙️ Installation  
Clone the repo and install dependencies:  

```bash
git clone https://github.com/<your-username>/Cephalometric-Landmark-Detection.git
cd Cephalometric-Landmark-Detection
pip install -r requirements.txt

---

## ⚙️ Usage  
- **Training**  
  ```bash
  python train.py --csv train.csv --images ./cepha400/ --epochs 50


