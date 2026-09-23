# 📊 Customer Churn Prediction Using ANN

A machine learning web application that predicts whether a bank customer is likely to churn using an Artificial Neural Network (ANN).

The application allows users to enter customer information and receive a predicted churn probability through an interactive Streamlit interface.

## 🚀 Live Demo

🔗 [Try the Customer Churn Prediction App](https://annclassificationchurn-xz2yd4gnax5hy3pja8ba29.streamlit.app/)

---

## 📌 Project Overview

Customer churn occurs when a customer stops using a company's products or services.

In the banking industry, predicting customer churn can help organizations identify customers who may need better engagement and retention strategies.

This project uses an Artificial Neural Network to analyze customer information and predict the probability of churn.

---

## 🎯 Objectives

- Predict customer churn using an ANN model.
- Perform data preprocessing and feature transformation.
- Use trained encoders and a scaler for model input.
- Develop an interactive web application using Streamlit.
- Deploy the application online using Streamlit Cloud.

---

## 🧠 Machine Learning Model

The project uses an Artificial Neural Network for binary classification.

### Input Features

The model uses the following customer attributes:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership Status
- Estimated Salary

### Prediction Output

The model generates a churn probability.

| Prediction Probability | Interpretation |
|---|---|
| Greater than 0.50 | Customer is predicted to churn |
| 0.50 or below | Customer is predicted not to churn |

The threshold is a chosen classification rule and does not guarantee the customer's future behavior.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

## ⚙️ Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Preprocessing
       ↓
Feature Encoding
       ↓
Feature Scaling
       ↓
ANN Model Training
       ↓
Model Evaluation
       ↓
Model Saving
       ↓
Streamlit Web Application
       ↓
Customer Churn Prediction
```

---

## 🖥️ Application Features

- Interactive customer input form
- Geography selection
- Gender selection
- Age and tenure controls
- Financial information input
- Churn probability display
- Churn prediction result
- Online deployment through Streamlit Cloud

---

## 📂 Project Structure

```text
ANN_Project/
│
├── app.py
├── model.h5
├── scaler.pkl
├── ohe_geo.pkl
├── label_encoder_gender.pkl
├── requirements.txt
└── README.md
```

---

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd ANN_Project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Model Preprocessing

The application uses the following preprocessing techniques:

- Label encoding for gender.
- One-hot encoding for geography.
- Feature scaling using StandardScaler.

The same preprocessing objects used during model training are loaded into the application to ensure consistent input transformation.

---

## 🌐 Deployment

The application is deployed using Streamlit Cloud.

Deployment process:

```text
GitHub Repository
       ↓
Streamlit Cloud
       ↓
Install Dependencies
       ↓
Load Model and Preprocessing Files
       ↓
Run Streamlit Application
```

---

## ⚠️ Limitations

- The prediction is based on patterns learned from the training dataset.
- The model's output is probabilistic and is not guaranteed to be accurate.
- Performance depends on the quality and representativeness of the training data.
- The application should not be used as the sole basis for financial or customer-related decisions.
- The classification threshold may require further validation.

---

## 🔮 Future Improvements

- Compare ANN performance with other machine learning models.
- Add confusion matrix and classification metrics.
- Improve model explainability using SHAP or other interpretability methods.
- Add model performance monitoring.
- Improve input validation.
- Add customer retention recommendations based on model insights.

---

## 👨‍💻 Author

** Mohd Ayan**

This project was developed as part of my machine learning learning journey to gain practical experience in ANN development, preprocessing, and application deployment.

---

## ⭐ Acknowledgements

Thanks to the open-source Python and machine learning community for providing the tools and libraries used in this project.
