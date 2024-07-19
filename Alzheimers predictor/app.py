import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model_path = 'C:\\Users\\~ideapadGAMING~\\Documents\\MyPrograms\\BootCamp\\Alzheimers predictor\\ML_MODEL\\My_decision_tree_model.pkl'
# model_path = 'C:\\Users\\~ideapadGAMING~\\Documents\\MyPrograms\\BootCamp\\Alzheimers predictor\\ML_MODEL\\My_LOGREG_model.pkl'
model = joblib.load(model_path)

try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    st.title('Alzheimer\'s Disease Prediction')

    st.write('Enter Information')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Patient Information')

        patient_name = st.text_input('Patient Name')
        age = st.slider('Age', 18, 100, 65)
        gender = st.selectbox('Gender', ['Female', 'Male'])
        ethnicity = st.selectbox('Ethnicity', ['Asian', 'Black', 'Hispanic', 'White'])
        education_level = st.selectbox('Education Level', ['High School', 'Some College', 'College Graduate', 'Post Graduate'])
        bmi = st.slider('BMI', 15.0, 40.0, 25.0)
        smoking = st.selectbox('Smoking', ['Never', 'Former', 'Current'])
        alcohol_consumption = st.slider('Alcohol Consumption (0-10)', 0, 10, 5)
        physical_activity = st.slider('Physical Activity (0-10)', 0, 10, 5)
        diet_quality = st.slider('Diet Quality (0-10)', 0, 10, 5)
        sleep_quality = st.slider('Sleep Quality (0-10)', 0, 10, 5)
        family_history_alzheimers = st.selectbox('Family History of Alzheimer\'s', ['No', 'Yes'])
        cardiovascular_disease = st.selectbox('Cardiovascular Disease', ['No', 'Yes'])
        diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
        depression = st.selectbox('Depression', ['No', 'Yes'])
        head_injury = st.selectbox('Head Injury', ['No', 'Yes'])
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
        systolic_bp = st.slider('Systolic Blood Pressure', 90, 200, 120)
        diastolic_bp = st.slider('Diastolic Blood Pressure', 60, 120, 80)
        cholesterol_total = st.slider('Total Cholesterol', 100, 300, 200)
        cholesterol_ldl = st.slider('LDL Cholesterol', 50, 200, 100)
        cholesterol_hdl = st.slider('HDL Cholesterol', 20, 100, 50)
        cholesterol_triglycerides = st.slider('Triglycerides', 50, 500, 150)
        mmse = st.slider('MMSE Score', 0, 30, 24)
        functional_assessment = st.slider('Functional Assessment (0-10)', 0, 10, 5)
        memory_complaints = st.selectbox('Memory Complaints', ['No', 'Yes'])
        behavioral_problems = st.selectbox('Behavioral Problems', ['No', 'Yes'])
        adl = st.slider('Activities of Daily Living (0-10)', 0, 10, 5)
        confusion = st.selectbox('Confusion', ['No', 'Yes'])
        disorientation = st.selectbox('Disorientation', ['No', 'Yes'])
        personality_changes = st.selectbox('Personality Changes', ['No', 'Yes'])
        difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', ['No', 'Yes'])
        forgetfulness = st.selectbox('Forgetfulness', ['No', 'Yes'])

    gender = 1 if gender == 'Female' else 0
    ethnicity = {'Asian': 0, 'Black': 1, 'Hispanic': 2, 'White': 3}.get(ethnicity, 0)
    education_level = {'High School': 0, 'Some College': 1, 'College Graduate': 2, 'Post Graduate': 3}.get(education_level, 0)
    smoking = {'Never': 0, 'Former': 1, 'Current': 2}.get(smoking, 0)
    family_history_alzheimers = 1 if family_history_alzheimers == 'Yes' else 0
    cardiovascular_disease = 1 if cardiovascular_disease == 'Yes' else 0
    diabetes = 1 if diabetes == 'Yes' else 0
    depression = 1 if depression == 'Yes' else 0
    head_injury = 1 if head_injury == 'Yes' else 0
    hypertension = 1 if hypertension == 'Yes' else 0
    memory_complaints = 1 if memory_complaints == 'Yes' else 0
    behavioral_problems = 1 if behavioral_problems == 'Yes' else 0
    confusion = 1 if confusion == 'Yes' else 0
    disorientation = 1 if disorientation == 'Yes' else 0
    personality_changes = 1 if personality_changes == 'Yes' else 0
    difficulty_completing_tasks = 1 if difficulty_completing_tasks == 'Yes' else 0
    forgetfulness = 1 if forgetfulness == 'Yes' else 0

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Ethnicity': [ethnicity],
        'EducationLevel': [education_level],
        'BMI': [bmi],
        'Smoking': [smoking],
        'AlcoholConsumption': [alcohol_consumption],
        'PhysicalActivity': [physical_activity],
        'DietQuality': [diet_quality],
        'SleepQuality': [sleep_quality],
        'FamilyHistoryAlzheimers': [family_history_alzheimers],
        'CardiovascularDisease': [cardiovascular_disease],
        'Diabetes': [diabetes],
        'Depression': [depression],
        'HeadInjury': [head_injury],
        'Hypertension': [hypertension],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'CholesterolTotal': [cholesterol_total],
        'CholesterolLDL': [cholesterol_ldl],
        'CholesterolHDL': [cholesterol_hdl],
        'CholesterolTriglycerides': [cholesterol_triglycerides],
        'MMSE': [mmse],
        'FunctionalAssessment': [functional_assessment],
        'MemoryComplaints': [memory_complaints],
        'BehavioralProblems': [behavioral_problems],
        'ADL': [adl],
        'Confusion': [confusion],
        'Disorientation': [disorientation],
        'PersonalityChanges': [personality_changes],
        'DifficultyCompletingTasks': [difficulty_completing_tasks],
        'Forgetfulness': [forgetfulness]
    })

    input_data = input_data[expected_columns]

    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            maxx=0.55
            minn=0.42
            probability = ((model.predict_proba(input_data)[0][1])-minn)/(maxx-minn)
           
            st.write(f'Prediction for {patient_name}: {"Alzheimer\'s Disease" if prediction[0] == 1 else "No Alzheimer\'s Disease"}')
            st.write(f'Probability of Alzheimer\'s Disease: {probability:.2f}')

            
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            sns.barplot(x=['No Alzheimer\'s Disease', 'Alzheimer\'s Disease'], y=[1 - probability, probability], ax=axes[0], palette=['green', 'red'])
            axes[0].set_title('Alzheimer\'s Disease Probability')
            axes[0].set_ylabel('Probability')

            sns.histplot(input_data['MMSE'], kde=True, ax=axes[1])
            axes[1].set_title('MMSE Score Distribution')

            axes[2].pie([1 - probability, probability], labels=['No Alzheimer\'s Disease', 'Alzheimer\'s Disease'], autopct='%1.1f%%', colors=['green', 'red'])
            axes[2].set_title('Alzheimer\'s Disease Pie Chart')

            st.pyplot(fig)

            if prediction[0] == 1:
                st.error(f"{patient_name} is likely to have Alzheimer's Disease. Consider seeking medical attention and following a healthy lifestyle.")
            else:
                st.success(f"{patient_name} is likely to not have Alzheimer's Disease. Keep up the good habits and maintain a healthy lifestyle.")

if __name__ == '__main__':
    main()
