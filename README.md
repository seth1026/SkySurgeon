# SkySurgeon - Predictive Maintenance (PdM) of Aircraft Engine using Machine Learning and Deep Learning

Predictive Maintenance techniques are used to determine the condition of equipment and predict potential failures before they occur. By forecasting failures in advance, organizations can significantly reduce equipment downtime and maintenance costs.

This project implements and evaluates multiple **machine learning and deep learning approaches** for predictive maintenance using **multivariate time-series data** from aircraft engines.

The project focuses on two main predictive tasks:

- **Classification:** Predict whether an engine will fail within the next *n* operational cycles.
- **Regression:** Predict the **Remaining Useful Life (RUL)** of an engine before failure.

---

# Dataset

The dataset used in this project is the **NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS)**.

Each dataset consists of **multivariate time-series sensor readings** collected from multiple engines operating under different conditions.

### Dataset Characteristics

- Each time series represents **one engine's lifecycle**
- Engines start in a **healthy operating state**
- A **fault develops gradually** until system failure
- Training data contains **complete run-to-failure sequences**
- Test data **ends before failure**, requiring RUL prediction

### Dataset Statistics

| Dataset | Engines | Cycle Length |
|-------|--------|--------------|
| Training Set | 100 | 128 – 356 cycles |
| Test Set | 100 | Ends before failure |

Dataset Link:  
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

---

# Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand sensor behavior and degradation patterns.

Key analysis steps include:

- Visualization of sensor trends across engine cycles
- Identification of **informative vs redundant sensors**
- Sensor data normalization and scaling
- Correlation analysis among sensors
- Remaining Useful Life (RUL) distribution analysis
- Detection of degradation patterns in time-series signals

---

# Predictive Maintenance Models

Multiple traditional and deep learning models were implemented and compared.

## Regression Models (RUL Prediction)

These models estimate the **Remaining Useful Life (RUL)** of an engine.

### 1. Exponential Degradation Model
Models engine degradation using an exponential decay function to estimate remaining life.

### 2. Similarity-Based Model
Predicts RUL by comparing current engine degradation trajectories with historical engine data.

### 3. LSTM Model
A deep learning sequence model capable of capturing **long-term temporal dependencies** in sensor time-series data.

---

## Classification Models (Failure Prediction)

These models predict whether an engine will fail within a defined time horizon.

### 4. LSTM (Binary & Multiclass Classification)
Detects early failure patterns using temporal dependencies in sensor data.

### 5. RNN (Binary & Multiclass Classification)
Uses recurrent neural networks to model sequential sensor behavior.

### 6. 1D CNN
Extracts temporal features from time-series sensor data using convolutional filters.

### 7. CNN-SVM Hybrid Model
Combines **CNN feature extraction** with **SVM classification** for binary failure prediction.

---

# Experimentation and Evaluation

Models were evaluated using appropriate metrics for regression and classification tasks.

## Regression Metrics

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- NASA RUL scoring function

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional experimentation included:

- Sliding window time-series generation
- Feature scaling and normalization
- Hyperparameter tuning
- Model comparison across architectures

---

# Future Work

Potential improvements and extensions include:

- **AutoKeras** for automated model discovery
- **Tsfresh** for automated time-series feature extraction
- **Dynamic Time Warping (DTW)** for time-series similarity and clustering
- **Genetic Algorithms** for feature selection and optimization
- **Hidden Markov Models (HMM)** for degradation state modeling
- **Survival Analysis** for probabilistic failure prediction
- **Autoencoders** for anomaly detection in sensor data

---

# Tech Stack

- Python
- TensorFlow / Keras
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# Project Structure

```
Predictive-Maintenance/
│
├── data/
│   ├── train/
│   └── test/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── RUL_models.ipynb
│   └── classification_models.ipynb
│
├── models/
│   ├── lstm.py
│   ├── cnn.py
│   └── svm.py
│
├── utils/
│   └── preprocessing.py
│
└── README.md
```

---

# References

- NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS)
- Predictive Maintenance Literature on Time-Series Machine Learning

