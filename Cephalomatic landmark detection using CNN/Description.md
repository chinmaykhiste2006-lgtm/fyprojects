# 🧠 Cephalometric Landmark Detection  

## 📌 Description  
An AI-based system for **automated cephalometric landmark detection** using deep learning.  
The model locates **19 anatomical landmarks** on cephalometric radiographs, improving efficiency, accuracy, and consistency in orthodontic analysis compared to manual identification.  

---

## 🚀 Features  
- 🔍 Detects 19 anatomical landmarks automatically  
- 🖼️ Preprocessing: image resizing + coordinate scaling  
- 🎛️ Augmentation: horizontal flips & brightness changes  
- 🧠 Model: ResNet18 backbone + regression head  
- 📊 Outputs (x, y) landmark coordinates  

---

## ⚙️ Installation  
```bash
git clone https://github.com/<your-username>/Cephalometric-Landmark-Detection.git
cd Cephalometric-Landmark-Detection
pip install -r requirements.txt

## Usage
**Training**
python train.py --csv train.csv --images ./cepha400/ --epochs 50
**Testing / Inference**
python test.py --csv test.csv --images ./cepha400/
Output
Input: Cephalometric X-ray
Output: Same image with 19 predicted landmarks
<p align="center"> <img src="assets/sample.png" alt="Sample Output" width="400"/> </p>
## 🧪 Results
✅ Detects 19 landmarks on cephalograms
🔄 Robust to flips and brightness changes
📏 Evaluation metric: Mean Euclidean Distance Error
<p align="center"> <img src="assets/results.png" alt="Results" width="500"/> </p>
✅ Future Enhancements
📈 Improve accuracy with deeper CNNs / transformers
🖥️ Add clinical visualization tools
🌐 Deploy as a web/desktop app for orthodontists
👨‍💻 Authors
Sharwill Kiran Khisti (Group Leader)
Chinmay Rajesh Khiste
Shraddha Prakash Khetmalis
Sairaj Ramesh Khot
Krishna Dinesh Khiraiya
Ritesh Vijay Khotale
