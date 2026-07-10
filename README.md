# Telco Customer Churn Prediction using ANN

A deep learning project that predicts whether a telecom customer is likely to churn using a feedforward Artificial Neural Network, specifically a Multi-Layer Perceptron (MLP), built with PyTorch.

## Project Overview

Customer churn is a major business problem for telecom companies. This project uses customer demographic information, service details, contract information, and billing features to predict whether a customer is likely to leave the company.

The project follows a complete deep learning workflow:

1. Load and understand the dataset
2. Clean the data
3. Perform exploratory data analysis
4. Encode categorical features
5. Scale numerical features
6. Convert data into PyTorch tensors
7. Train an ANN/MLP model
8. Evaluate using classification metrics
9. Improve the model using class weighting, dropout, and weight decay
10. Tune the prediction threshold

## Dataset

Dataset used: **Telco Customer Churn Dataset**

Expected file name:

```text
Telco-Customer-Churn.csv
```

Place the dataset file inside:

```text
elco-Customer-Churn.csv
```

The dataset contains customer-level information such as gender, senior citizen status, tenure, phone service, internet service, contract type, payment method, monthly charges, total charges, and churn status.

## Problem Type

This is a **binary classification** problem.

Target column:

```text
Churn
```

Target values:

```text
No  -> 0
Yes -> 1
```

## Model Used

The project uses a feedforward Artificial Neural Network:

```text
Artificial Neural Network
└── Feedforward Neural Network
    └── Multi-Layer Perceptron
```

Model architecture:

```text
Input features -> 64 neurons -> 32 neurons -> 16 neurons -> 1 output
```

The model uses:

- Fully connected layers
- ReLU activation functions
- Dropout for regularization
- BCEWithLogitsLoss for binary classification
- Adam optimizer
- Class weighting to handle imbalance
- Threshold tuning for better churn detection

## Results

### Basic MLP

| Metric | Score |
|---|---:|
| Accuracy | 0.7839 |
| Precision | 0.6122 |
| Recall | 0.5107 |
| F1 Score | 0.5569 |

### Improved MLP

| Metric | Score |
|---|---:|
| Accuracy | 0.7392 |
| Precision | 0.5062 |
| Recall | 0.7674 |
| F1 Score | 0.6100 |

### Threshold Tuning

| Threshold | Accuracy | Precision | Recall | F1 Score |
|---:|---:|---:|---:|---:|
| 0.3 | 0.6510 | 0.4257 | 0.8957 | 0.5771 |
| 0.4 | 0.6915 | 0.4565 | 0.8422 | 0.5921 |
| 0.5 | 0.7392 | 0.5062 | 0.7674 | 0.6100 |
| 0.6 | 0.7591 | 0.5345 | 0.7246 | 0.6152 |
| 0.7 | 0.7896 | 0.6102 | 0.5775 | 0.5934 |

Final threshold selected:

```text
0.6
```

because it achieved the best F1-score.

## Key Findings from EDA

- Customers with month-to-month contracts have much higher churn.
- Fiber optic customers show higher churn compared to DSL customers.
- Customers using electronic check have the highest churn rate among payment methods.
- Tenure has a negative relationship with churn.
- Monthly charges have a positive relationship with churn.
- Accuracy alone is not enough because the dataset is imbalanced.

## Project Structure

```text
telco-churn-ann-pytorch/
├── notebook/
│   ├── customer-churn-prediction
├── dataset/
│   ├── Telco-customer-churn
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```



## Important Learning Points

This project demonstrates:

- Difference between classification and regression
- Why preprocessing is important
- Why categorical columns must be encoded
- Why numerical columns should be scaled
- How DataLoader works in PyTorch
- How an ANN/MLP is built
- Why BCEWithLogitsLoss is used for binary classification
- Why recall and F1-score are important for imbalanced data
- How class weighting improves recall
- How threshold tuning affects precision and recall

## Final Conclusion

The improved MLP model is preferred because it significantly improves recall for churn customers. In a customer churn problem, recall is important because the business wants to identify as many potential churn customers as possible before they leave. Although the improved model has slightly lower accuracy, it provides better business value by detecting more high-risk customers.

## Future Improvements

- Compare ANN with Logistic Regression, Random Forest, and XGBoost
- Add validation set and early stopping
- Perform hyperparameter tuning
- Use SHAP for model explainability
- Build a Streamlit web app
- Deploy the model using FastAPI
