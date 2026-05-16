
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import sklearn
print(sklearn.__version__)


# For deployment on Streamlit.io or similar platforms, you should place
# your model files (joblib files) in the same directory as this script
# or a subdirectory within your deployment repository.

# Define the base path for the joblib files as the current directory.
# Ensure 'modelo pred churn.joblib' and 'preprocessor_ohe.joblib' are in this directory.
base_path = '.' # Assuming models are in the same directory as EcRapp.py

# Load the models and preprocessors
try:
    churn_model = joblib.load(os.path.join(base_path, 'modelo pred churn.joblib'))
    preprocessor_ohe = joblib.load(os.path.join(base_path, 'preprocessor_ohe.joblib'))
    st.success("Modelos y preprocesadores cargados correctamente.")
except Exception as e:
    st.error(f"Error cargando modelos o preprocesadores: {e}")
    st.stop() # Stop the app if models cannot be loaded

st.title('Eco Ride: Predicción de Churn')
st.write('Ingrese los datos del cliente para predecir si dejará el servicio (Churn).')

# Input fields for numerical features
uso_mensual_km = st.number_input('Uso Mensual (Km)', min_value=0.0, value=1000.0, step=100.0)
soporte_tickets = st.number_input('Tickets de Soporte', min_value=0, value=2, step=1)
dias_antiguedad = st.number_input('Días de Antigüedad', min_value=0, value=365, step=1)

# Input field for categorical feature (Region)
region_options = ['Centro', 'Norte', 'Sur']
selected_region = st.selectbox('Región', region_options)

if st.button('Predecir Churn'):
    # Create a single DataFrame with all raw inputs, ensuring correct column names and order
    input_data = pd.DataFrame([[uso_mensual_km, soporte_tickets, dias_antiguedad, selected_region]],
                              columns=['Uso_Mensual_Km', 'Soporte_Tickets', 'Dias_Antiguedad', 'Region'])

    # Apply the preprocessor_ohe (assuming it's a ColumnTransformer that handles both numerical scaling and OHE)
    processed_features = preprocessor_ohe.transform(input_data)

    # Get the feature names after transformation from the preprocessor_ohe
    final_feature_names = preprocessor_ohe.get_feature_names_out()
    
    # Strip the prefixes added by ColumnTransformer (e.g., 'cat__' and 'remainder__')
    # to match the feature names the model expects.
    cleaned_feature_names = [col.split('__', 1)[-1] if '__' in col else col for col in final_feature_names]
    
    final_features_df = pd.DataFrame(processed_features, columns=cleaned_feature_names)

    # Make prediction
    prediction = churn_model.predict(final_features_df)
    prediction_proba = churn_model.predict_proba(final_features_df)

    st.subheader('Resultado de la Predicción:')
    if prediction[0] == 1:
        st.error('El cliente **probablemente dejará el servicio (Churn)**.')
    else:
        st.success('El cliente **probablemente NO dejará el servicio (No Churn)**.')

    st.write(f"Probabilidad de Churn: {prediction_proba[0][1]:.2f}")
    st.write(f"Probabilidad de No Churn: {prediction_proba[0][0]:.2f}")
