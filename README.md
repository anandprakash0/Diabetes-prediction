🩺 Diabetes Prediction using Machine Learning

![Diabetes Prediction](https://img.shields.io/badge/Machine%20Learning-Diabetes%20Prediction-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

This project aims to predict the likelihood of an individual having diabetes based on various health-related attributes. Utilizing machine learning algorithms, the model analyzes input data to provide accurate predictions, aiding in early diagnosis and management of diabetes.

## 📂 Project Structure

Diabetes-prediction/
├── PIMA.ipynb
├── dataset_prediction.ipynb
├── diabetes.ipynb
├── diabetes_012_health_indicators_BRFSS2015.csv
├── diabetes_prediction_dataset.csv
├── pima-data.csv
├── LICENSE
└── README.md


## 🧠 Algorithms Implemented

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

## 📊 Dataset Description

The project utilizes the following datasets:

1. **PIMA Indians Diabetes Dataset**: Medical data of Pima Indian women aged 21 and above.
2. **BRFSS 2015 Dataset**: Behavioral Risk Factor Surveillance System data focused on health-related risk behaviors.
3. **Custom Diabetes Prediction Dataset**: Curated dataset combining various health indicators.

Each dataset includes features such as:

- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0: Non-diabetic, 1: Diabetic)

## 🛠️ Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anandprakash0/Diabetes-prediction.git
   cd Diabetes-prediction

    Create a virtual environment (optional):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt

If requirements.txt is not available, install dependencies manually:

pip install pandas numpy matplotlib seaborn scikit-learn

Run the Jupyter Notebook:

    jupyter notebook

    Open any of the .ipynb files to start exploring the analysis and models.

📈 Model Evaluation Metrics

    Accuracy Score

    Confusion Matrix

    Precision, Recall, F1-Score

    ROC-AUC Curve

🔍 Exploratory Data Analysis (EDA)

Included in the notebooks:

    Missing value treatment

    Outlier detection

    Feature scaling

    Correlation analysis

    Data visualization

🚀 Future Enhancements

    Add deep learning models (e.g., Neural Networks)

    Deploy with Flask or Streamlit

    Use more datasets for greater diversity

    Add cross-validation and hyperparameter tuning

🤝 Contributing

Contributions are welcome! Fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.
📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
