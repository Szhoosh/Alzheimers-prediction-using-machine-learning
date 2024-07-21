# Alzheimer's Disease Prediction App

This Streamlit app predicts the likelihood of Alzheimer's Disease based on patient information and a trained machine learning model.

## Features

- User interface to input patient data.
- Prediction of Alzheimer's Disease with probability.
- Visualizations of prediction results.

## Installation

Requirements:
  Python 3.7 or higher
  Streamlit
  pandas
  joblib
  matplotlib
  seaborn

1. Clone the repository:

    ```bash
    git clone https://github.com/Szhoosh/Alzheimers-prediction-using-machine-learning.git

    cd Alzheimers-prediction-using-machine-learning
    ```

2. Open cmd prompt

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Place your trained model file in the specified directory (if not already present):
    ```plaintext
    C:\PathToYourDirectory\Alzheimers-prediction-using-machine-learning\ML_MODEL\My_decision_tree_model.pkl
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the app.

3. Fill in the patient information and click on the `Predict` button to see the results.


## File Structure


## File Structure

```plaintext
alzheimers-prediction-app/
├── app.py
├── requirements.txt
├── README.md
├── Data_sets/
│   ├── alzheimers_disease_data.csv
│   └── filtered_data.csv
└── ML_MODEL/
    ├── My_decision_tree_model.pkl
    └── My_LOGREG_model.pkl
