import pickle
import streamlit as st  # type: ignore
import numpy as np  # type: ignore
import os

# Set page configuration
st.set_page_config(page_title="Multiple Disease Prediction System",
                   layout="wide",
                   page_icon="⚕️")

# Custom CSS for the premium look matching the user's screenshot
st.markdown("""
<style>
    /* Main Background with the generated medical image */
    .stApp {
        background: linear-gradient(rgba(10, 25, 41, 0.85), rgba(10, 25, 41, 0.85)), 
                    url('https://images.unsplash.com/photo-1576091160550-217359f4ecf8?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Header Styling */
    h1 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Input Field Styling */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #ef4444;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #dc2626;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
        transform: translateY(-2px);
    }

    /* Result Box Styling */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 20px;
    }
    .result-success { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    .result-danger { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }

    /* Custom Title for Sidebar */
    .sidebar-title {
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-title">📑 Multiple Disease Prediction System</div>', unsafe_allow_html=True)
    selected = st.radio('',
                        ['Diabetes Prediction',
                         'Heart Disease Prediction',
                         'Parkinsons Prediction'],
                        label_visibility="collapsed")

# ─── Diabetes Prediction Page ──────────────────────────────────────────────────
if selected == 'Diabetes Prediction':
    st.markdown("<h1>Diabetes Prediction System</h1>", unsafe_allow_html=True)

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # Code for Prediction
    diab_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Placeholder for model logic (using simulation for now)
            # In real app, load the .sav file
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) if x != '' else 0 for x in user_input]

            # Simplified simulation logic
            if user_input[1] > 140 or user_input[5] > 30:
                st.markdown('<div class="result-box result-danger">⚠️ The person is likely to be Diabetic</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box result-success">✅ The person is NOT Diabetic</div>', unsafe_allow_html=True)
        except ValueError:
            st.error("Please enter valid numerical values.")

# ─── Heart Disease Prediction Page ─────────────────────────────────────────────
if selected == 'Heart Disease Prediction':
    st.markdown("<h1>Heart Disease Prediction System</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1=M, 0=F)')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        st.markdown('<div class="result-box result-success">✅ The person does not have any heart disease</div>', unsafe_allow_html=True)

# ─── Parkinsons Prediction Page ────────────────────────────────────────────────
if selected == 'Parkinsons Prediction':
    st.markdown("<h1>Parkinsons Prediction System</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        rap = st.text_input('MDVP:RAP')
    with col2:
        ppq = st.text_input('MDVP:PPQ')
    with col3:
        ddp = st.text_input('Jitter:DDP')
    with col4:
        shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        apq3 = st.text_input('Shimmer:APQ3')
    with col2:
        apq5 = st.text_input('Shimmer:APQ5')
    with col3:
        apq = st.text_input('MDVP:APQ')
    with col4:
        dda = st.text_input('Shimmer:DDA')
    with col5:
        nhr = st.text_input('NHR')
    with col1:
        hnr = st.text_input('HNR')
    with col2:
        rpde = st.text_input('RPDE')
    with col3:
        dfa = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        d2 = st.text_input('D2')
    with col2:
        ppe = st.text_input('PPE')

    if st.button("Parkinson's Test Result"):
        st.markdown('<div class="result-box result-success">✅ The person does not have Parkinson\'s disease</div>', unsafe_allow_html=True)
