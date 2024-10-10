import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_random_forest_model2.joblib')
# Define our top 3 features expected by the model
features = [
    'Cloud_Cover_Percentage','Mean_Temp','Precipitation_Amount']

def process_input(input_data):
    """
    Preprocess the input data if needed.
    For now, it's returning the input as-is.
    """
    return input_data

# Streamlit app setup
st.title('🌍 Lumpy Skin Disease Prediction 🐄')
st.write("Use this app to predict the likelihood of Lumpy Skin Disease based on environmental factors.")

# Sidebar for instructions
st.sidebar.header('🔍 Instructions')
st.sidebar.info(
    'Fill in the environmental data in the fields below. '
    'Once ready, click the **Predict** button to get the prediction results.'
)

# Input section header
st.header('Enter Environmental Data')

# Input fields for all features
input_data = {}
for feature in features:
    # Create numeric input fields for each feature with a description
    input_data[feature] = st.number_input(
        f'Enter {feature}', 
        format='%.6f', 
        help=f'Provide the value for {feature} in relevant units.'
    )

# Button for making predictions
if st.button('🔮 Predict Now!'):
    # Convert user inputs into a DataFrame for model prediction
    input_df = pd.DataFrame([input_data])
    
    # Process the input (in case you have transformations)
    processed_input = process_input(input_df)
    
    # Make prediction and get probabilities
    prediction_proba = model.predict_proba(processed_input)[0]  # Get probabilities
    prediction = model.predict(processed_input)[0]  # Get final prediction (0 or 1)
    
    # Display the prediction result
    if prediction == 1:
        st.success(f'✅ The predicted class is: Positive (Lumpy Skin Disease Detected)')
        st.info(f'Probability of Positive: {prediction_proba[1]:.4f}')
        st.info(f'Probability of Negative: {prediction_proba[0]:.4f}')
    else:
        st.success(f'❌ The predicted class is: Negative (No Lumpy Skin Disease)')
        st.info(f'Probability of Negative: {prediction_proba[0]:.4f}')
        st.info(f'Probability of Positive: {prediction_proba[1]:.4f}')
    
    # Optionally, display the processed input for transparency or debugging
    st.write("📊 Processed Input Data (for debugging):")
    st.write(processed_input)

# Footer section in the sidebar
st.sidebar.header('ℹ️ About')
st.sidebar.info(
    'This application predicts the likelihood of Lumpy Skin Disease in cattle '
    'based on various environmental factors like cloud cover, precipitation, and temperature. '
    'It utilizes a LightGBM model trained on historical data.'
)

# A footer with styling tips and colors
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <style>
    .css-1aumxhk {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        font-size: 16px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)
