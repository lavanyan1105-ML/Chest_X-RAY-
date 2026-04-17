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

> ⚠️ Note: The test dataset is kept constant across all experiments and never used for training.

---

## ⚙️ Technologies Used

* Python
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* OpenCV

---

## 🧠 Models Implemented

### 🔹 1. Support Vector Machine (SVM)

* Kernel: Linear
* Feature Extraction:

  * Statistical features
  * Histogram-based features
* Pipeline:

  * Scaling → PCA → SVM

---

### 🔹 2. Convolutional Neural Network (CNN)

* Custom deep CNN architecture
* Layers:

  * Conv2D + BatchNorm + MaxPooling
  * Dropout (to prevent overfitting)
* Trained for **100+ epochs**
* Callbacks:

  * EarlyStopping
  * ModelCheckpoint
  * ReduceLROnPlateau

---

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

---

## 🏆 Best Model

👉 **Custom CNN / Transfer Learning model** performed best overall due to:

* Ability to capture spatial features
* Better generalization on image data

---

## 📊 Visualizations Included

* Class distribution histogram
* Training curves (loss & accuracy)
* Confusion matrices
* Accuracy vs Training Size plots
* Model comparison charts

---

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

## 🚀 How to Run

### ▶️ Google Colab

1. Upload dataset to Colab
2. Open notebook
3. Run all cells sequentially

### ▶️ Local Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost opencv-python
```

---

## 📌 Key Learnings

* Deep learning models outperform traditional ML for image tasks
* Transfer learning reduces training time and improves accuracy
* More data → better performance (especially for CNNs)
* Feature extraction is critical for SVM/XGBoost

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Use advanced architectures (ResNet, EfficientNet)
* Handle class imbalance
* Deploy model using Flask/Streamlit

---

## 🙌 Acknowledgements

* Kaggle Chest X-ray Dataset
* TensorFlow & Scikit-learn documentation
* XGBoost community

---

## 📬 Contact

For queries or collaboration:

* GitHub: *your-username*
* Email: *your-email*

---
