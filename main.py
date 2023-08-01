import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title('Kidney Disease Prediction Application')
st.write('''
         Please fill in the attributes below, then hit the Predict button
         to get your results. 
         ''')

st.header('Input Attributes')
age = st.slider('Your Age (Years)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
st.write(''' ''')

bp = st.slider('Blood Pressure (mm/Hg)', min_value=0.0, max_value=200.0, value=150.0, step=1.0)
st.write(''' ''')
s = st.radio("Specific Gravity (SG)", ('SG 1.005: Very Low Urnine Concentration', 'SG 1.010: Moderately Low Urnine Concentration', 'SG 1.015: Normal', 'SG 1.020: Slightly High Urine Concentration','SG 1.025: High Urine Concentration'))
st.write(''' ''')
# Specific Gravity
if s == "SG 1.005: Very Low Urnine Concentration":
    sg = 1.005
elif s == "SG 1.010: Moderately Low Urnine Concentration":
    sg = 1.010
elif s == "SG 1.015: Normal":
    sg = 1.015
elif s == "SG 1.020: Slightly High Urine Concentration":
    sg = 1.020
else:
    sg = 1.025


a = st.radio("Albumin Level (g/L)", ('Low (less then 33.9)', 'Slightly Low (33.9-35)', 'Normal (35 – 50 g/L)', 'Slightly High (50 - 51.5)', 'High (51.5 - 150)' , 'Extremely High (Over 150)'))
st.write(''' ''')
# Specific Gravity
if a == "Low (less then 33.9)":
    al = 0
elif a == "Slightly Low (33.9-35)":
    al = 1
elif a == "Normal (35 – 50 g/L)":
    al = 2
elif a == "Slightly High (50 - 51.5)":
    al = 3
elif a == "High (51.5 - 100)":
    al = 4
else:
    al = 5

sug = st.radio("Sugar Level", ('Low', 'Slightly Low', 'Normal', 'Slightly High', 'High' , 'Extremely High'))
st.write(''' ''')
# Specific Gravity
if sug == "Low)":
    sugar = 0
elif sug == "Slightly Low":
    sugar = 1
elif sug == "Normal":
    sugar = 2
elif sug == "Slightly High":
    sugar = 3
elif sug == "High":
    sugar = 4
else:
    sugar = 5

red = st.radio("Red Blood Cell Count", ('Normal', 'Abnormal'))
st.write(''' ''')
# blood cell
if red == "Normal":
    rbc = 0
else:
    rbc = 1

pus = st.radio("Pus Cell Count", ('Normal', 'Abnormal'))
st.write(''' ''')
# pus cell
if pus == "Normal":
    pc = 0
else:
    pc = 1

pusc = st.radio("Pus Cell Clumps", ('Present', 'Not Present'))
st.write(''' ''')
# pus cell
if pusc == "Present":
    pcc = 1
else:
    pcc = 0

ba = st.radio("Bacterial Infection", ('Present', 'Not Present'))
st.write(''' ''')
# pus cell
if ba == "Present":
    bac = 1
else:
    bac = 0

bgr = st.slider('Blood Glucose Random (mgs/dl)', min_value=0.0, max_value=600.0, value=300.0, step=1.0)
st.write(''' ''')
bu = st.slider('Blood Urea (mgs/dl)', min_value=0.0, max_value=500.0, value=250.0, step=0.1)
st.write(''' ''')
sc = st.slider('Serum Creatinine (mgs/dl)', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
st.write(''' ''')
sod = st.slider('Sodium (mEq/L)', min_value=0.0, max_value=200.0, value=100.0, step=0.1)
st.write(''' ''')
pot = st.slider('Potassium (mEq/L)', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
st.write(''' ''')
hemo = st.slider('Hemoglobin (gms)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
st.write(''' ''')
pcv = st.slider('Packed  Cell Volume', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
st.write(''' ''')
wbc = st.slider('White Blood Cell Count (cells/cumm)', min_value=0.0, max_value=50000.0, value=25000.0, step=1.0)
st.write(''' ''')
rbcc = st.slider('Red Blood Cell Count (millions/cmm)', min_value=0.0, max_value=200.0, value=100.0, step=1.0)
st.write(''' ''')
hyp = st.radio("Hypertension", ('Yes', 'No'))
st.write(''' ''')
if hyp == "Yes":
    htn = 1
else:
    htn = 0

diam = st.radio("Diabetes Mellitus", ('Yes', 'No'))
st.write(''' ''')
if diam == "Yes":
    dm = 1
else:
    dm = 0

cor = st.radio("Coronary Artery Disease", ('Yes', 'No'))
st.write(''' ''')
if cor == "Yes":
    cad = 1
else:
    cad = 0

app = st.radio("Appetite", ('Good', 'Poor'))
st.write(''' ''')
if app == "Good":
    appet = 1
else:
    appet = 0

pedal = st.radio("Pedal Edema", ('Yes', 'No'))
st.write(''' ''')
if pedal == "Yes":
    pe = 1
else:
    pe = 0

anemia = st.radio("Anemia", ('Yes', 'No'))
st.write(''' ''')
if anemia == "Yes":
    ane = 1
else:
    ane = 0



selected_models = st.multiselect("Choose Classifier Models", ('Random Forest', 'Naïve Bayes', 'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'LightGBM', 'XGBoost', 'Multilayer Perceptron', 'Artificial Neural Network', 'Support Vector Machine'))
st.write(''' ''')

# Initialize an empty list to store the selected models
models_to_run = []

# Check which models were selected and add them to the models_to_run list
if 'Random Forest' in selected_models:
    models_to_run.append(RandomForestClassifier())

if 'Naïve Bayes' in selected_models:
    models_to_run.append(GaussianNB())

if 'Logistic Regression' in selected_models:
    models_to_run.append(LogisticRegression())


if 'Decision Tree' in selected_models:
    models_to_run.append(DecisionTreeClassifier())

if 'Gradient Boosting' in selected_models:
    models_to_run.append(GradientBoostingClassifier())

if 'Support Vector Machine' in selected_models:
    models_to_run.append(SVC())

if 'LightGBM' in selected_models:
    models_to_run.append(LGBMClassifier())

if 'XGBoost' in selected_models:
    models_to_run.append(XGBClassifier())

if 'Multilayer Perceptron' in selected_models:
    models_to_run.append(MLPClassifier())

if 'Artificial Neural Network' in selected_models:
    models_to_run.append(MLPClassifier(hidden_layer_sizes=(100,), max_iter=100))


user_input = np.array([age, bp, sg, al, sugar, rbc, pc, pcc, bac, bgr, bu, sc,
                       sod, pot, hemo, pcv, wbc, rbcc, htn, dm, cad, appet, pe, ane]).reshape(1, -1)

# import dataset
def get_dataset():
    data = pd.read_csv('kidney.csv')

    # Calculate the correlation matrix
    # corr_matrix = data.corr()

    # Create a heatmap of the correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.title('Correlation Matrix')
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=0)
    # plt.tight_layout()

    # Display the heatmap in Streamlit
    # st.pyplot()

    return data

def generate_model_labels(model_names):
    model_labels = []
    for name in model_names:
        words = name.split()
        if len(words) > 1:
            # Multiple words, use initials
            label = "".join(word[0] for word in words)
        else:
            # Single word, take the first 3 letters
            label = name[:3]
        model_labels.append(label)
    return model_labels

if st.button('Submit'):
    df = get_dataset()

    # fix column names
    df.columns = (["id", "age", "bp", "sg", "al", "su", "rbc", "pc",
                   "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
                   "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane", "class"])

    # Transforming classification into numerical format
    df['class'] = df['class'].apply(lambda x: 1 if x == 'ckd' else 0)

    # Transforming ane into numerical format
    df['ane'] = df['ane'].apply(lambda x: 1 if x == 'yes' else 0)

    # Transforming pe into numerical format
    df['pe'] = df['pe'].apply(lambda x: 1 if x == 'yes' else 0)

    # Transforming appet into numerical format
    df['appet'] = df['appet'].apply(lambda x: 1 if x == 'poor' else 0)

    # Transforming cad into numerical format
    df['cad'] = df['cad'].apply(lambda x: 1 if x == 'yes' else 0)

    # Transforming dm into numerical format
    df['dm'] = df['dm'].apply(lambda x: 1 if x == 'yes' else 0)

    # Transforming htn into numerical format
    df['htn'] = df['htn'].apply(lambda x: 1 if x == 'yes' else 0)

    # Transforming ba into numerical format
    df['ba'] = df['ba'].apply(lambda x: 1 if x == 'present' else 0)

    # Transforming pcc into numerical format
    df['pcc'] = df['pcc'].apply(lambda x: 1 if x == 'present' else 0)

    # Transforming pc into numerical format
    df['pc'] = df['pc'].apply(lambda x: 1 if x == 'abnormal' else 0)

    # Transforming rbc into numerical format
    df['rbc'] = df['rbc'].apply(lambda x: 1 if x == 'abnormal' else 0)


    # Replace NaN values with median for float columns
    float_columns = df.select_dtypes(include=['float']).columns
    df[float_columns] = df[float_columns].fillna(df[float_columns].median())

    # Convert columns to numeric
    numeric_columns = ['pcv', 'wc', 'rc']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Replace NaN values with median for numeric columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())


    # Split the dataset into train and test
    X = df.drop(['class','id'], axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create two columns to divide the screen
    left_column, right_column = st.columns(2)


    # Left column content
    with left_column:
        # Create a VotingClassifier with the top 3 models
        ensemble = VotingClassifier(
            estimators=[('rf', RandomForestClassifier()), ('xgb', XGBClassifier()), ('gb', GradientBoostingClassifier())],
            voting='hard')

        # Fit the voting classifier to the training data
        ensemble.fit(X_train, y_train)

        # Make predictions on the test set
        model_predictions = ensemble.predict(user_input)

        # Evaluate the model's performance on the test set
        ensamble_accuracy = accuracy_score(y_test, ensemble.predict(X_test))
        ensamble_precision = precision_score(y_test, ensemble.predict(X_test))
        ensamble_recall = recall_score(y_test, ensemble.predict(X_test))
        ensamble_f1score = f1_score(y_test, ensemble.predict(X_test))

        if model_predictions == 1:
            st.write(f'According to Ensemble Model You have a **Very High Chance (1)** of Kidney Disease.')
        else:
            st.write(f'According to Ensemble Model You have a **Very Low Chance (0)** of Kidney Disease.')

        st.write('Ensemble Model Accuracy:', ensamble_accuracy)
        st.write('Ensemble Model Precision:', ensamble_precision)
        st.write('Ensemble Model Recall:', ensamble_recall)
        st.write('Ensemble Model F1 Score:', ensamble_f1score)
        st.write('------------------------------------------------------------------------------------------------------')


    # Right column content
    with right_column:

        for model in models_to_run:
            # Train the selected model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            model_predictions = model.predict(user_input)

            # Evaluate the model's performance on the test set
            model_accuracy = accuracy_score(y_test, model.predict(X_test))
            model_precision = precision_score(y_test, model.predict(X_test))
            model_recall = recall_score(y_test, model.predict(X_test))
            model_f1score = f1_score(y_test, model.predict(X_test))

            if model_predictions == 1:
                st.write(f'According to {type(model).__name__} Model You have a **Very High Chance (1)** of Kidney Disease.')
            else:
                st.write(f'According to {type(model).__name__} Model You have a **Very Low Chance (0)** of Kidney Disease.')

            st.write(f'{type(model).__name__} Accuracy:', model_accuracy)
            st.write(f'{type(model).__name__} Precision:', model_precision)
            st.write(f'{type(model).__name__} Recall:', model_recall)
            st.write(f'{type(model).__name__} F1 Score:', model_f1score)
            st.write('------------------------------------------------------------------------------------------------------')

    # Initialize lists to store model names and their respective performance metrics
    model_names = ['Ensemble']
    accuracies = [ensamble_accuracy]
    precisions = [ensamble_precision]
    recalls = [ensamble_recall]
    f1_scores = [ensamble_f1score]

    # Loop through the selected models to compute their performance metrics
    for model in models_to_run:
        model_names.append(type(model).__name__)
        model.fit(X_train, y_train)
        model_predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, model_predictions))
        precisions.append(precision_score(y_test, model_predictions))
        recalls.append(recall_score(y_test, model_predictions))
        f1_scores.append(f1_score(y_test, model_predictions))

    # Create a DataFrame to store the performance metrics
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    # Get the model labels
    model_labels = generate_model_labels(metrics_df['Model'])

    # Plot the comparison graphs
    plt.figure(figsize=(12, 10))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    plt.bar(model_labels, metrics_df['Accuracy'], color='skyblue')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)

    # Precision comparison
    plt.subplot(2, 2, 2)
    plt.bar(model_labels, metrics_df['Precision'], color='orange')
    plt.title('Precision Comparison')
    plt.ylim(0, 1)

    # Recall comparison
    plt.subplot(2, 2, 3)
    plt.bar(model_labels, metrics_df['Recall'], color='green')
    plt.title('Recall Comparison')
    plt.ylim(0, 1)

    # F1 Score comparison
    plt.subplot(2, 2, 4)
    plt.bar(model_labels, metrics_df['F1 Score'], color='purple')
    plt.title('F1 Score Comparison')
    plt.ylim(0, 1)

    # Adjust layout to prevent overlapping of titles
    plt.tight_layout()

    # Display the graphs in Streamlit
    st.pyplot()
