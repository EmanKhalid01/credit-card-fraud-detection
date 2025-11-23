# Credit Card Fraud Detection Using Machine Learning
A machine learning project that detects fraudulent credit card transactions using the highly imbalanced Kaggle dataset.
SMOTE is used to balance the data, and Random Forest (with hyperparameter tuning) is used to build a high-performance classifier.
# 🚀 Project Overview
This project aims to build a robust fraud detection system that can help identify potential fraudulent transactions:
 # Key Features
- Complete Data preprocessing
- Handling missing values
- Exploratory Data Analysis (EDA)
- Feature Scaling
- SMOTE oversampling
- Random Forest Classifier model
- Hyperparameter tuning
- Model saving using joblib
- Full evaluation metrics: Accuracy, Precision, Recall, F1 Score, etc.
# 📂 Project Structure
  credit-card-fraud-detection/
- ├── fraud_detection.ipynb
- ├── fraud_model.pkl (ignored)
- ├── requirements.txt
- ├── README.md
- └── venv/ (ignored)
- └── creditcard.csv (ignored)
# 🧠 Machine Learning Workflow
- Load the dataset
- Data Cleaning
- Exploratory Data Analysis
- Train–Test Split
- SMOTE oversampling
- Random Forest training
- Hyperparameter tuning
- Model evaluation
- Save the final model
# 📊 Model Performance
  Best Model:  Random Forest + SMOTE + Hyperparameter Tuning
- Accuracy: 99.92%
- Recall (Fraud class): 86.70%
- Precision (Fraud class): 74.56%
- F1-Score: 80.18%.
<br> These metrics indicate a strong ability to detect and prevent fraudulent transactions.
# 🔧 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
# 👇 How to Run This Project
 - Clone the repo:
   git clone https://github.com/<your-username>/credit-card-fraud-detection.git
- Install dependencies:
  pip install -r requirements.txt
- Open the notebook:
  jupyter notebook
- Run all cells.
# 💾 Saving & Loading the Model
 # Save model
joblib.dump(best_model, "fraud_model.pkl")
 # Load model
model = joblib.load("fraud_model.pkl")
# 📌 Dataset Link
Kaggle Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
# ✨ Author
Eman Khalid
- GitHub: https://github.com/EmanKhalid01
- LinkedIn: https://linkedin.com/in/eman-khalid001

