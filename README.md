# 😷 Real-Time Face Mask Detector

A **Computer Vision & Deep Learning project** that detects whether a person is wearing a **face mask or not in real-time** using a webcam.  
The system uses **Transfer Learning with VGG16** and **OpenCV** for real-time face detection and classification.

---

## 🚀 Project Overview

This project combines **deep learning** with **live video processing** to build an intelligent system capable of detecting face masks in real time.

The model uses **VGG16 pre-trained architecture** to extract features and classify faces into two categories:

- ✅ **Mask**
- ❌ **No Mask**

Faces are first detected using **Haar Cascade Classifier**, then passed to the trained deep learning model for prediction.

---

## 🧠 Tech Stack

- **Python**
- **OpenCV**
- **TensorFlow / Keras**
- **VGG16 (Transfer Learning)**
- **Haar Cascade Classifier**
- **NumPy**

---

## 📊 Model Performance

| Metric | Score |
|------|------|
| Training Accuracy | 99% |
| Validation Accuracy | **98%** |

The use of **transfer learning** significantly improved accuracy while reducing training time.

---

## ⚙️ How It Works

1️⃣ Webcam captures live video frames  
2️⃣ **OpenCV Haar Cascade** detects faces in each frame  
3️⃣ Detected faces are **preprocessed and resized**  
4️⃣ The trained **VGG16-based CNN model** predicts mask or no mask  
5️⃣ Bounding box + prediction label is displayed in real-time

---

## ▶️ Run Real-Time Detection

```bash
python detect_mask_video.py
```

Your webcam will start and the system will **detect masks in real-time**.

---

## 🧪 Training the Model

To train the model from scratch:

```bash
python train_mask_detector.py
```



---

## 📸 Example Output

- Green Box → **Mask Detected**
- Red Box → **No Mask Detected**

---

## 🎯 Applications

- Public safety monitoring
- Smart surveillance systems
- Healthcare environments
- Airport & transport security



