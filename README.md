# 🛩️ Aerial Object Classification & Detection  
### Bird vs Drone | Custom CNN • Transfer Learning • YOLOv8 • Streamlit Deployment

---

## 📌 Project Overview  
This project focuses on **classifying aerial images as Bird or Drone** and optionally **detecting objects using YOLOv8**.  
Such models are essential in:

- 🛡️ Security & Defense Airspace Monitoring  
- 🐦 Wildlife & Environmental Research  
- ✈️ Airport Bird-Strike Prevention  
- 🚁 Unauthorized Drone Detection  

The project includes:

- Custom CNN model  
- Transfer learning (MobileNetV2)  
- YOLOv8 Object Detection  
- Streamlit App for deployment  
- Fully documented workflow  

---

## 📁 Project Structure

- Aerial_Object_Classification_Detection/
- │
- ├── classification_dataset/
- │ ├── train/bird
- │ ├── train/drone
- │ ├── valid/bird
- │ ├── valid/drone
- │ └── test/bird, drone
- │
- ├── object_detection_dataset/
- │ ├── train/images, train/labels
- │ ├── valid/images, valid/labels
- │ ├── test/images, test/labels
- │ └── data.yaml
- │
- ├── src/
- │   ├── 1_preprocessing.py
- │   ├── 2_custom_cnn_train.py
- │   ├── 3_transfer_learning_train.py
- │   ├── 4_evaluate_models.py
- │   ├── 5_yolov8_train.py
- │   └── utils.py
- |
- ├── notebooks/
- │ ├── Classification_Training.ipynb
- │ └── YOLOv8_Training.ipynb
- │
- ├── models/
- │ ├── custom_cnn.h5
- │ ├── mobilenet_best.h5
- │ └── yolov8n-best.pt
- │
- ├── streamlit_app/
- │ ├── app.py
- │ └── requirements.txt
- │
- └── README.md


---

## 📌 Features

### 🔹 1. Image Classification  
- Binary classification (Bird / Drone)  
- Custom CNN architecture  
- Transfer Learning (MobileNetV2)

### 🔹 2. Object Detection  
- YOLOv8 Nano model  
- Bounding box visualization  
- Real-time inference

### 🔹 3. Deployment  
- Streamlit interactive UI  
- Upload → Predict → Visualize  

---


## 🧠 Model Comparison

| Model | Accuracy | F1 Score | Params | Notes |
|-------|----------|----------|--------|-------|
| Custom CNN | ~92% | High | Low | Fast, lightweight |
| MobileNetV2 | ~97% | Very High | Medium | Best performer |
| YOLOv8n | Detects Bird/Drone | — | Low | Real-time detection |

---

## 🎯 Business Applications

- Detect drones entering restricted airspace  
- Protect aircraft from bird-strikes  
- Monitor wildlife migration  
- Identify drones used for illegal surveillance  

---

## 🧩 Tech Stack  
**Deep Learning | Computer Vision | TensorFlow | MobileNet | YOLOv8 | Streamlit | Python**

---



