import pickle
import streamlit as st
import pandas as pd

# Load the trained Isolation Forest model
with open('iso_forest_model.pkl', 'rb') as model_file:
    iso_forest = pickle.load(model_file)

# Load the scaler (if you have one)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title and description
st.title("Credit Card Fraud Detection App")
st.write("""
This app detects fraudulent credit card transactions using a trained anomaly detection model.
Upload a new dataset to see the results.
""")

# File uploader for users to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

def detect_anomalies(new_data, model, scaler):
    # Preprocess the new data (e.g., scaling Amount and Time)
    if 'Amount' in new_data.columns and 'Time' in new_data.columns:
        new_data[['Amount', 'Time']] = scaler.transform(new_data[['Amount', 'Time']])
    
    # Predict anomalies using the trained model
    anomalies = model.predict(new_data)
    
    # Convert -1 to 1 (fraud) and 1 to 0 (non-fraud)
    anomalies = [1 if x == -1 else 0 for x in anomalies]
    
    # Add the anomalies as a new column in the dataset
    new_data['Anomaly'] = anomalies
    
    return new_data

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        new_transactions = pd.read_csv(uploaded_file)
        
        # Display the uploaded data
        st.write("Uploaded Data")
        st.write(new_transactions.head())

        # Preprocess the new data and detect anomalies using the trained model (e.g., Isolation Forest)
        new_transactions_with_anomalies = detect_anomalies(new_transactions, iso_forest, scaler)
        
        # Display the transactions classified as fraudulent
        st.write("Fraudulent Transactions Detected")
        st.write(new_transactions_with_anomalies[new_transactions_with_anomalies['Anomaly'] == 1])
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Upload a CSV file to detect fraud!")

# Running the app
if __name__ == '__main__':
    st.write("Upload a CSV file to detect fraud!")
