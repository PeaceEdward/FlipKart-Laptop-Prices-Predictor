import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

file_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "laptop.csv"))
le_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "encoder.pkl"))
model_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "rfmodel.pkl"))
le_os_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "encoder2.pkl"))
le_brand_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "encoder3.pkl"))
le_ramtype_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "encoder.pkl4"))

df=pd.read_csv(file_path)


with open(model_path, 'rb') as f:
    rf_model = pickle.load(f)

with open(le_path, 'rb') as f:
    le_processor = pickle.load(f)
    
with open(le_os_path, 'rb') as f:
    le_os= pickle.load(f)
    
with open(le_brand_path, 'rb') as f:
    le_brand = pickle.load(f)
    
with open(le_ramtype_path, 'rb') as f:
    le_ramtype = pickle.load(f)


st.title('Laptop Price Predictor')

# Transform categorical features using LabelEncoder()
le_processor = LabelEncoder().fit(df['Processor'])
le_os = LabelEncoder().fit(df['OS'])
le_brand = LabelEncoder().fit(df['Brand'])
le_ramtype = LabelEncoder().fit(df['RAMType'])

# Get user input
brand = st.selectbox('Brand', df['Brand'].unique())
os = st.selectbox('Operating System', df['OS'].unique())
processor = st.selectbox('Processor', df['Processor'].unique())
ram_type = st.selectbox('RAM Type', df['RAMType'].unique())
ram_sizes = np.sort(df['RAMSize'].unique())
ram_size = st.selectbox('RAM Size (in GB)', ram_sizes)
ssd_sizes = np.sort(df['Storage_SSD'].unique())
ssd_size = st.selectbox('SSD Size (in GB)', ssd_sizes)
hdd_sizes = np.sort(df['Storage_HDD'].unique())
hdd_size = st.selectbox('HDD Size (in GB)', hdd_sizes)

# Encode user input using the fitted LabelEncoder()
processor_encoded = le_processor.transform([processor])[0]
os_encoded = le_os.transform([os])[0]
brand_encoded = le_brand.transform([brand])[0]
ram_type_encoded = le_ramtype.transform([ram_type])[0]

# Make a prediction
features = [processor_encoded, os_encoded, brand_encoded, ram_type_encoded, rating, display, ram_size, hdd_size, ssd_size]
final_features = np.array(features).reshape(1, -1)
prediction = rf_model.predict(final_features)

# Display the prediction
st.subheader('Prediction')    
