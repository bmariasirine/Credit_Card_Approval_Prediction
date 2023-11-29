# Credit Card Approval Prediction

## Overview
This project focuses on predicting credit card approval using machine learning techniques. It employs the XGBoost algorithm to analyze applicant data, preprocess features, and make predictions. The model achieves 68.36% accuracy on the test set.

## Features
- Data preprocessing: Handles missing values, encodes categorical variables using Label Encoding.
- Model training: Utilizes XGBoost classifier for credit card approval prediction.
- Evaluation: Assesses model accuracy on the test set.

## Dataset
- Two datasets used: `application_record.csv` and `credit_record.csv`.
- Merged on 'ID' for comprehensive analysis.

## Usage
1. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn xgboost matplotlib`.
2. Adjust file paths and features in the code based on your dataset.
3. Run the code to train the model and visualize feature importances.

## Colab Notebook
Explore the code and results in this interactive [Colab Notebook](https://colab.research.google.com/drive/1zYU7riB2qdNYmGpWyj5-Ty5IsRK0DN1c?usp=sharing).

## Files
- `credit_approval_prediction.py`: Python file containing the code.
- `application_record.csv` and `credit_record.csv`: Input datasets.
- `requirements.txt`: List of required Python packages.

## Results
- Model achieves 68.36% accuracy.
- Feature importances plotted for better interpretability.

## Contributors
- Bekka Maria Sirine

Feel free to contribute and open issues for improvements!
