# Kidney Disease Prediction Model using Ensemble Learning
This repository contains a kidney disease prediction model built using ensemble learning techniques. The model aims to predict whether a patient is likely to have kidney disease based on various input attributes. The model has achieved an impressive 96% accuracy on the test dataset.

Live Demo: [Kidney Disease Prediction](https://kidney-disease-prediction-using-ensemble.streamlit.app/)

# Dataset
The model is trained and tested on a kidney disease dataset (kidney.csv) that contains various features related to a patient's health and laboratory test results. The dataset is preprocessed to handle missing values and transformed into a format suitable for model training.

# Input Attributes
The prediction application allows users to input the following attributes:

  1. Age (Years): The age of the patient.
  2. Blood Pressure (mm/Hg): The patient's blood pressure.
  3. Specific Gravity (SG): The specific gravity of urine, indicating urine concentration.
  4. Albumin Level (g/L): The level of albumin in the blood, which can indicate kidney health.
  5. Sugar Level: The level of sugar in the urine.
  6. Red Blood Cell Count: Whether the red blood cell count is normal or abnormal.
  7. Pus Cell Count: Whether the pus cell count is normal or abnormal.
  8. Pus Cell Clumps: Whether pus cell clumps are present or not.
  9. Bacterial Infection: Whether a bacterial infection is present or not.
  10. Blood Glucose Random (mgs/dl): Random blood glucose level.
  11. Blood Urea (mgs/dl): Blood urea level.
  12. Serum Creatinine (mgs/dl): Serum creatinine level.
  13. Sodium (mEq/L): Sodium level in the blood.
  14. Potassium (mEq/L): Potassium level in the blood.
  15. Hemoglobin (gms): Hemoglobin level.
  16. Packed Cell Volume: Packed cell volume level.
  17. White Blood Cell Count (cells/cumm): White blood cell count.
  18. Red Blood Cell Count (millions/cmm): Red blood cell count.
  19. Hypertension: Whether the patient has hypertension or not.
  20. Diabetes Mellitus: Whether the patient has diabetes mellitus or not.
  21. Coronary Artery Disease: Whether the patient has coronary artery disease or not.
  22. Appetite: The patient's appetite (good or poor).
  23. Pedal Edema: Whether the patient has pedal edema or not.
  24. Anemia: Whether the patient has anemia or not.
  
# Ensemble Learning Models
The kidney disease prediction model is implemented using several ensemble learning algorithms to improve the prediction accuracy. The user can choose one or more models for comparison from the following list:

  1. Random Forest
  2. Naïve Bayes
  3. Logistic Regression
  4. Decision Tree
  5. Gradient Boosting
  6. Support Vector Machine
  7. LightGBM
  8. XGBoost
  9. Multilayer Perceptron (MLP)
  10. Artificial Neural Network (ANN)

# Screenshot
![Screenshot (869)](https://github.com/SaifSunny/Kidney-Disease-Prediction-using-Ensemble/assets/72490093/64fb844a-4399-4ec8-91d9-be2f1eb8bd20)
![Screenshot (871)![Screenshot (872)](https://github.com/SaifSunny/Kidney-Disease-Prediction-using-Ensemble/assets/72490093/86eeca3e-679f-4ef6-bb5e-e38469279cb0)
](https://github.com/SaifSunny/Kidney-Disease-Prediction-using-Ensemble/assets/72490093/edccccfc-e9f3-4260-a60d-1e1d70ca4ab2)

# How to Use
1. Visit the live demo link: [Kidney Disease Prediction](https://kidney-disease-prediction-using-ensemble.streamlit.app/).
2. Fill in the required input attributes as described above.
3. Click on the "Submit" button to get predictions from the ensemble model and the selected individual models.
4. The application will display the prediction results and performance metrics of each model, including accuracy, precision, recall, and F1 score.
5. Users can select different models to compare their performance with the ensemble model.

# Data Source
The kidney disease dataset used for training and testing the model is not included in this repository. You can find publicly available kidney disease datasets from various sources, or you can use your own dataset following the same format as described in the Input Attributes section.

# Disclaimer
The predictions made by the kidney disease prediction model are based on the available data and the performance of the chosen models. The model's accuracy is not a guarantee of real-world performance, and it is essential to consult a medical professional for a reliable diagnosis. The authors and contributors of this repository are not responsible for any decisions or actions made based on the predictions generated by the model.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
