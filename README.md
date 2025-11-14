# Demand Forecasting Using Temporal Fusion Transformers (TFT)

This project implements a **Temporal Fusion Transformer (TFT)** model for **multivariate time-series forecasting**, specifically designed for **retail demand prediction**. The TFT architecture combines attention mechanisms, gating layers, and variable selection networks to deliver accurate, interpretable, and scalable forecasting for real-world business applications.

---

## 🚀 Project Overview

The goal of this project is to build a robust forecasting system capable of predicting future retail demand using historical sales data along with external covariates such as holidays, promotions, and weather conditions.

The model leverages:

- **Temporal Fusion Transformers (TFT)** for long- and short-term temporal pattern learning  
- **Attention-based interpretability** for understanding feature importance  
- **Optuna** for hyperparameter optimization  
- **Advanced feature engineering** to incorporate time-based and external factors  

This approach achieves **superior accuracy (low MAE/RMSE)** and enables businesses to make informed decisions for inventory, pricing, and supply chain planning.

---

## 🧰 Tech Stack

- **Python**
- **PyTorch Forecasting**
- **TensorFlow**
- **Optuna**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**

---

## 📦 Features

### ✔ Multivariate time-series forecasting  
Handles multiple covariates and historical variables.

### ✔ Built on Temporal Fusion Transformers (TFT)  
State-of-the-art architecture with:

- Variable selection networks  
- Gated residual networks  
- Temporal attention  
- Static & dynamic feature modeling  

### ✔ Advanced Feature Engineering  
Includes:

- Time-related features (day, week, month, seasonality)  
- Holidays & special events  
- Promotional flags  
- Weather data integration  

### ✔ Hyperparameter Optimization  
Automated with **Optuna** for:

- Learning rate  
- Hidden size  
- Dropout  
- Batch size  
- Sequence lengths  

### ✔ Interpretability  
Attention visualization allows users to:

- Understand influential time periods  
- Identify key features affecting demand  
- Support strategic business decisions  

---


