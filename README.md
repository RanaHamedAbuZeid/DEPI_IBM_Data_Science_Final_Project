# DEPI_IBM_Data_Science_Final_Project
# Data science project on Lumpy Skin Disease - Using Machine Learning (Classification)

## Project Overview

  This repository contains the final project files for the **Lumpy Skin Disease Forecasting** project. The project applies machine learning techniques to predict the occurrence of **Lumpy Skin Disease** based on meteorological and geospatial features. The goal is to build models using Random Forest and XGBoost classifiers, handle imbalanced data, and perform feature selection to improve predictive accuracy.

## Project Team members 

- 1- Rana Hamed
- 2- Heba Ibrahim
- 3- Beshoy Ageeb 
- 4- Eman Mansour

## Project Structure

The repository contains the following important files:

- **`presentation.pdf`**: A comprehensive presentation that outlines the entire data science life cycle followed during the project, including data preprocessing, model building, evaluation, and insights. It highlights key methodologies, challenges faced, solutions implemented, and final outcomes in a visually engaging format for easy understanding by both technical and non-technical audiences.

- **`ML_report.docx`**: This document contains the detailed machine learning report, including model evaluation, performance metrics, and hyperparameter tuning results.
  
- **`EDA_report.docx`**: This document includes the exploratory data analysis (EDA) and data preprocessing steps performed on the dataset.

- **`Model building.ipynb`**: This Jupyter notebook contains the complete code for building the machine learning models, from importing the dataset to evaluating the model performance.

- **`preprocessed_data.csv`**: This CSV file contains the cleaned and preprocessed data used in model training and evaluation.

- **`EDA_file.ipynb`**: This Jupyter notebook contains the code for EDA and data preprocessing, including handling missing values, feature engineering, and data visualization.

- **`Lumpy skin disease data.csv`**: This CSV file contains the original dataset before preprocessing.

- **`ne_110m_admin_0_countries.dbf`**: An auxiliary file that might be related to geographical data used in EDA or visualization (specific usage can be explained further if required).
  
- **`Dashboard.py`** file contains the implementation of a Python dashboard that allows users to interact with many visualizations, and explore the results. 
---
- **`streamlit.py`**: file contains the implementation of a **Streamlit** that allows users to interact with the trained machine learning models, show the predictions, and explore the results. The website provides an intuitive interface for non-technical users to understand the model performance.
- or use this streamlet link directly: https://lumpydetection.streamlit.app/
-  **`best_random_forest_model2.joblib`** file contains the final model used in the streamlit.


---

## Machine Learning Process

### Step 1: Importing Required Libraries
The essential libraries for data handling, machine learning, feature selection, and evaluation metrics are imported.

### Step 2: Data Import and Preprocessing
- The preprocessed dataset (`preprocessed_data.csv`) is loaded.
- Certain columns like `Longitude` and `Latitude` are dropped based on the analysis.
- **Imbalanced Data Handling**: We use **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the imbalanced target variable by oversampling the minority class.

### Step 3: Feature Selection
- We utilize **SelectKBest** based on the ANOVA F-test to select the most significant features for classification, reducing the dimensionality to focus on the top 3 features.

### Step 4: Data Scaling
- Feature scaling is performed using **StandardScaler** to normalize the feature values before feeding them into the model.

### Step 5: Model Building and Hyperparameter Tuning
- **Random Forest** and **XGBoost** models are built, and their hyperparameters are tuned using **GridSearchCV** to find the optimal set of parameters.

### Step 6: Model Validation and Testing
- Both models are validated on a separate validation set.
- The models are then tested on an independent test set, and their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
  
### Step 7: Comparison of Models
- After testing, we compare the performance of Random Forest and XGBoost based on accuracy scores and confusion matrices to determine the better-performing model on test data.

### Step 8: Error Analysis (Optional)
- Misclassifications are analyzed using confusion matrices for both Random Forest and XGBoost models, providing insights into where the models may have underperformed.

---

## Code Overview

The main notebook (`Model building.ipynb`) contains the following key steps:

1. **Data Import and Preprocessing**: Handling missing values, feature selection, and handling class imbalance using SMOTE.
2. **Model Building**: Construction of the Random Forest and XGBoost classifiers.
3. **Hyperparameter Tuning**: Using `GridSearchCV` to find the best parameters.
4. **Model Evaluation**: Performance metrics like accuracy, precision, recall, F1-score, and confusion matrix.
5. **Feature Selection**: Utilizing ANOVA F-test for feature selection to reduce dimensionality.
6. **Comparison**: Comparing the two models' performance and selecting the best model based on the results.
---

## Conclusion

The project successfully demonstrates how to handle imbalanced data, perform feature selection, and build accurate machine-learning models. Both Random Forest and XGBoost classifiers were evaluated, with performance comparison highlighting the stronger model. Detailed reports and code files are provided for further reference and documentation.
---
---
## Dataset Reference

- **Kaggle Dataset**: [Lumpy Skin Disease Dataset](https://www.kaggle.com/datasets/saurabhshahane/lumpy-skin-disease-dataset/data)
- **Mendeley Data**: Afshari Safavi, Ehsanallah (2021), “Lumpy Skin disease dataset”, Mendeley Data, V1, doi: [10.17632/7pyhbzb2n9.1](https://doi.org/10.17632/7pyhbzb2n9.1)


