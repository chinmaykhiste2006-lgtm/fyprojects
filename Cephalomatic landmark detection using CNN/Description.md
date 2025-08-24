

```markdown
# 🧠 Cephalometric Landmark Detection  

## 📌 Project Overview  
This project focuses on **automatic cephalometric landmark detection** using deep learning.  
We developed a convolutional neural network (CNN) model based on **ResNet18** to detect 19 anatomical landmarks (38 coordinate values) from cephalometric X-ray images.  

The system is designed to assist in orthodontic and craniofacial analysis by automating landmark identification, which is usually a time-consuming manual task for experts.  

---

## 🚀 Features  
- Pretrained **ResNet18 backbone** for feature extraction.  
- Custom regression head for predicting **(x, y)** landmark coordinates.  
- **Adaptive scaling** of landmarks from original resolution to 512×512 input size.  
- **Augmentation techniques**: horizontal flip and brightness variation.  
- **PyTorch dataset pipeline** for efficient training and validation.  

---

## 📂 Dataset  
- Dataset consists of cephalometric X-ray images (`.png/.jpg`).  
- Landmarks are provided in a CSV file with structure:  
```

image\_path, x1, y1, x2, y2, ... , x19, y19

````
- Images are resized to **512×512** for training.  
- Landmarks are automatically scaled to match resized images.  

---

## 🏗️ Model Architecture  
1. **Backbone:** ResNet18 pretrained on ImageNet (last classification layers removed).  
2. **Global Average Pooling:** reduces feature maps to vector form.  
3. **Regression Head:** fully connected layers to map features → landmark coordinates.  

Output shape:  
- `19 × 2 = 38` values (x, y coordinates).  

---

## ⚙️ Installation  
Clone the repository:  
```bash
git clone https://github.com/your-username/cephalometric-landmark-detection.git
cd cephalometric-landmark-detection
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Training the Model

```bash
python train.py --csv train.csv --images cepha400/ --epochs 50 --batch_size 8
```

### Testing / Evaluation

```bash
python test.py --csv test.csv --images cepha400/ --weights model.pth
```

---

## 📊 Example Output

* Input: Cephalometric X-ray (512×512)
* Output: 19 predicted landmarks (visualized as plotted points).

---

## 👥 Authors

* **Sharwill Kiran Khisti (Group Leader)** – Lead Developer & Integration (implemented core model, coordinated dataset handling, and overall project flow).
* **Chinmay Rajesh Khiste** – Model Development (built CNN using ResNet18 backbone and designed regression head for landmark detection).
* **Shraddha Prakash Khetmalis** – Dataset Preparation (handled CSV parsing, preprocessing of cephalometric images, scaling of landmarks).
* **Sairaj Ramesh Khot** – Data Augmentation (implemented augmentations like flipping, brightness, and normalization for training).
* **Krishna Dinesh Khiraiya** – Training & Evaluation (managed training loops, validation, and testing landmark accuracy).
* **Ritesh Vijay Khotale** – Documentation & Testing (prepared project documentation, debugged dataset/model issues, and performed testing).

---





