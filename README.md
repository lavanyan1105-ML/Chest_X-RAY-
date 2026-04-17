# 🩺 Chest X-Ray Classification using SVM, CNN & XGBoost

## 📌 Project Overview

This project focuses on classifying chest X-ray images into **Normal** and **Pneumonia** categories using multiple machine learning and deep learning approaches.

The goal is to compare traditional ML models with deep learning and boosting techniques, and analyze how performance changes with different training data sizes.

---

## 🎯 Objectives

* Implement **SVM (Support Vector Machine)** for image classification
* Build a **Custom CNN model**
* Apply **Transfer Learning (VGG16 / MobileNetV2)**
* Implement **XGBoost-based models**
* Perform **training size experiments (20%, 40%, 60%, 80%)**
* Evaluate models using:

  * Accuracy
  * Precision
  * Recall (Sensitivity)
  * Specificity
  * F1-score

---

## 📂 Dataset

* **Chest X-ray Dataset**

  * Classes:

    * Normal
    * Pneumonia
* Dataset split:

  * Training Set
  * Validation Set
  * Test Set

<img width="806" height="609" alt="image" src="https://github.com/user-attachments/assets/21560e29-417d-4fe3-8ce9-b309994a64a4" />

<img width="726" height="490" alt="image" src="https://github.com/user-attachments/assets/e542ad77-9dda-4e24-bd2d-51a485445cc7" />
<img width="721" height="237" alt="image" src="https://github.com/user-attachments/assets/73f937f8-38c1-4ac5-9ae8-e11641d79ba6" />

## 🧠 Models Implemented

### 🔹 1. Support Vector Machine (SVM)
<img width="670" height="698" alt="image" src="https://github.com/user-attachments/assets/527ea8ef-2168-4ca3-88a2-465a1f9e5eec" />

* Kernel: Linear
* Feature Extraction:

  * Statistical features
  * Histogram-based features
* Pipeline:

  * Scaling → PCA → SVM

---<img width="738" height="708" alt="image" src="https://github.com/user-attachments/assets/6937ce64-4e44-4827-ad51-d468d2cd206b" />


### 🔹 2. Convolutional Neural Network (CNN)
<img width="771" height="334" alt="image" src="https://github.com/user-attachments/assets/e2f9044a-6583-497c-a468-fb4ec02d957d" />

* Custom deep CNN architecture
* Layers:

  * Conv2D + BatchNorm + MaxPooling
  * Dropout (to prevent overfitting)
* Trained for **100+ epochs**
* Callbacks:

  * EarlyStopping
  * ModelCheckpoint
  * ReduceLROnPlateau

<img width="712" height="616" alt="image" src="https://github.com/user-attachments/assets/880dafb0-81a4-4e27-b21b-77205e2ce40b" />


### 🔹 3. Transfer Learning

* Pretrained models:

  * VGG16 / MobileNetV2
* Frozen base layers
* Custom classification head added

---

### 🔹 4. XGBoost Approaches

* Raw XGBoost (flattened features)
* PCA + XGBoost
* VGG16 + XGBoost (deep feature extraction)

---

## 📊 Evaluation Metrics

All models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall (Sensitivity)**
* **Specificity**
* **F1-score**

Specificity is calculated using confusion matrix:

```
Specificity = TN / (TN + FP)
```

---

## 📈 Experiments

### 🔸 Training Size Analysis

Models were trained using:

* 20% of data
* 40% of data
* 60% of data
* 80% of data

### 🔸 Observations

* CNN performance improves significantly with more data
* SVM shows limited improvement
* XGBoost performs moderately but improves with feature engineering
* Transfer learning provides strong performance even with smaller data

---

## 📉 Results Summary

| Model             | Accuracy | F1-Score | Sensitivity | Specificity |
| ----------------- | -------- | -------- | ----------- | ----------- |
| SVM (Linear)      | ~0.65    | ~0.72    | ~0.73       | ~0.52       |
| Custom CNN        | ~0.87    | ~0.90    | ~0.88       | ~0.87       |
| Transfer Learning | ~0.90+   | ~0.91+   | High        | High        |
| PCA + XGBoost     | ~0.75    | ~0.83    | High        | Low         |
| VGG16 + XGBoost   | ~0.76    | ~0.84    | High        | Moderate    |
| Raw XGBoost       | ~0.73    | ~0.82    | High        | Low         |


## 📁 Project Structure

```
├── data/
│   ├── train/
│   ├── test/
│   └── val/
│
├── models/
│   ├── cnn_model.h5
│   ├── svm_model.joblib
│   ├── xgboost_model.json
│
├── results/
│   ├── metrics.csv
│   ├── plots/
│
├── notebook.ipynb
└── README.md
```

---

