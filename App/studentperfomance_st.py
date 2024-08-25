import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Function to preprocess input data
def preprocess_input(input_data, feature_order, scaler):
    input_df = pd.DataFrame(input_data)
    input_df = input_df[feature_order]  # Ensure the correct order of features
    input_scaled = scaler.transform(input_df)
    return input_scaled

# Function to train the model and prepare the scaler
def train_model_and_scaler(df):
    # Define the feature order
    feature_order = [
        'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 
        'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
        'Sports', 'Music', 'Volunteering', 'Age', 'GPA'
    ]
    
    # Feature Scaling
    scaler = StandardScaler()
    df[feature_order] = scaler.fit_transform(df[feature_order])

    X = df[feature_order]
    y = df['GradeClass']

    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=50)
    model.fit(X, y)
    
    return model, scaler, feature_order

# Initialize session state for storing inserted data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        'Name', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 
        'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
        'Sports', 'Music', 'Volunteering', 'Age', 'GPA'
    ])

# Grade mapping
grade_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'F'
}

# Streamlit Application
st.title('Student Grade Prediction App')

st.write("""
### Predict student grades based on survey responses
Fill in the survey beside to predict your potential grade class.
""")

# Sidebar Header and Survey Form
st.sidebar.header('Insert Data')

# Survey Form in Sidebar
name = st.sidebar.text_input("What's your name?")
gender = st.sidebar.selectbox("What is your gender?", [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
ethnicity = st.sidebar.selectbox("What is your ethnicity?", [0, 1, 2, 3], format_func=lambda x: ['Caucasian', 'African American', 'Asian', 'Other'][x])
parental_education = st.sidebar.selectbox("What is your parental education level?", [0, 1, 2, 3, 4], format_func=lambda x: ['None', 'High School', 'Some College', "Bachelor's", 'Higher'][x])
study_time = st.sidebar.slider("How many hours do you study weekly?", 0, 20, 10)
absences = st.sidebar.slider("How many absences do you have?", 0, 30, 0)
tutoring = st.sidebar.selectbox("Do you have tutoring?", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
parental_support = st.sidebar.selectbox("What is the level of parental support?", [0, 1, 2, 3, 4], format_func=lambda x: ['None', 'Low', 'Moderate', 'High', 'Very High'][x])
extracurricular = st.sidebar.selectbox("Are you involved in extracurricular activities?", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
sports = st.sidebar.selectbox("Do you participate in sports?", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
music = st.sidebar.selectbox("Are you involved in music activities?", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
volunteering = st.sidebar.selectbox("Do you do volunteering?", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
age = st.sidebar.slider("What is your age?", 15, 18, 17)
gpa = st.sidebar.slider("What is your GPA?", 2.0, 4.0, 3.0)

# Prepare the input data
input_data = {
    "Name": name,
    "Gender": gender,
    "Ethnicity": ethnicity,
    "ParentalEducation": parental_education,
    "StudyTimeWeekly": study_time,
    "Absences": absences,
    "Tutoring": tutoring,
    "ParentalSupport": parental_support,
    "Extracurricular": extracurricular,
    "Sports": sports,
    "Music": music,
    "Volunteering": volunteering,
    "Age": age,
    "GPA": gpa
}

# Insert data button
if st.sidebar.button('Insert Data'):
    if name.strip() == '':
        st.sidebar.write("Please enter your name.")
    else:
        new_data = pd.DataFrame([input_data])
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
        st.sidebar.write("Data inserted successfully!")

# Display inserted data in the main page
st.write("### Inserted Data")
st.dataframe(st.session_state.data)

# Load and preprocess data
df = pd.read_csv('Student.csv')

# Train the model and prepare the scaler and feature order
model, scaler, feature_order = train_model_and_scaler(df)

# Prediction
if st.button('Predict'):
    if not st.session_state.data.empty:
        # Predict for all inserted data
        data_to_predict = st.session_state.data.drop(columns=['Name'])
        data_scaled = preprocess_input(data_to_predict.to_dict(orient='records'), feature_order, scaler)
        predictions = model.predict(data_scaled)
        
        # Add predictions to the data
        st.session_state.data['GradeClass'] = predictions
        st.session_state.data['Grade'] = st.session_state.data['GradeClass'].map(grade_mapping)
        
        # Display results in attractive cards
        st.write("### Prediction Results")
        for index, row in st.session_state.data.iterrows():
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px; 
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                <h4>{row['Name']}</h4>
                <p><strong>Grade Class:</strong> {row['Grade']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No data available for prediction.")
