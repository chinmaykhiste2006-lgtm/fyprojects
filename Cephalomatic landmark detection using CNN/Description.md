

```markdown
# 🧠 Cephalometric Landmark Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen.svg)  
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## Project Overview
Automated **Cephalometric Landmark Detection** using **deep learning**. The system predicts **19 anatomical landmarks** on lateral skull X-rays for applications in **orthodontics, craniofacial surgery planning, and diagnostic analysis**.

Key components:
- PyTorch **training & evaluation pipeline**
- **Mean Euclidean Error (MEE)** computation in pixels and millimeters
- Overlay visualization of **predicted vs ground truth landmarks**
- Interactive **Streamlit web app** for real-time predictions

---

## Features
- Pretrained **ResNet18 backbone** fine-tuned for landmark regression
- **Custom PyTorch Dataset** with resizing and augmentation
- Augmentations: horizontal flip, brightness adjustment
- Training with **MSELoss** and validation with **MEE tracking**
- Evaluation script generating **overlay images and metrics**
- **Streamlit app** with visualization and CSV export

---

## Repository Structure
```

Cephalometric-Landmark-Detection/
│── dataset\_loader.py       # Custom PyTorch Dataset
│── model.py                # ResNet18-based CNN model
│── train.py                # Training script
│── evaluate.py             # Evaluation + visualization
│── app.py                  # Streamlit web app
│── utils.py                # Helper functions
│── best\_model.pth          # Saved model weights
│── mee\_score.txt           # Evaluation results
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
│
├── Cephalometric dataset/
│   ├── cepha400/           # X-ray images
│   ├── train\_senior.csv    # Training CSV
│   ├── test1\_senior.csv    # Validation CSV
│   ├── test2\_senior.csv    # Test CSV
│
├── output\_images/          # Landmark overlay images

```

---

## Dataset
- **Source**: Public cephalometric dataset (400+ lateral skull X-rays)
- **Annotations**: CSV files with image paths + **19 (x,y) landmark coordinates**
- **Preprocessing**:
  - Images resized to **512×512**
  - Landmarks scaled accordingly
  - Augmentations applied during training

---

## Model Architecture
- **Backbone**: ResNet18 pretrained on ImageNet
- **Regression Head**:
```

Flatten → Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → 38)

````
- **Output**: 38 values → 19 landmarks (x,y)

---

## Training
Run:
```bash
python train.py
````

* **Loss**: MSELoss
* **Optimizer**: Adam (lr=0.001)
* **Batch Size**: 32
* **Epochs**: 50

Sample training log:

```
Epoch 10/50 | Train Loss: 0.0021 | Val Loss: 0.0030 | MEE: 2.15 px / 0.57 mm
✅ Best model saved!
```

---

## Evaluation

Run:

```bash
python evaluate.py
```

* Computes **per-image and average MEE**
* Saves **overlay images** in `output_images/`
* Final score saved in `mee_score.txt`

---

## Streamlit Web App

Run:

```bash
streamlit run app.py
```

### Features

* Upload cephalometric X-ray → get predicted landmarks
* Toggle overlays:

  * Ground truth landmarks
  * Predicted landmarks
  * Error arrows
  * Landmark indices
* Plots:

  * Landmark overlay
  * Mean Euclidean Error (MEE)
  * Intensity histogram
  * Quadrant distribution
* Download predictions as **CSV**

---

## Results

* Achieved **Mean Euclidean Error (MEE)**:

```
X.XX px  ≈  Y.YY mm
```

* Predicted landmarks show strong alignment with ground truth

---

## Future Scope

* Heatmap-based landmark detection
* Transformer-based architectures for global context
* Semi-supervised learning with unlabeled data
* Larger datasets for improved generalization

---

## Team Members

* Member 1 – Model Development
* Member 2 – Data Preprocessing
* Member 3 – Web App Development
* Member 4 – Training & Evaluation
* Member 5 – Visualization & Documentation
* Member 6 – Project Management

```

---


```
