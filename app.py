import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model

yolo_model = YOLO("saved_models/best.pt")
lstm_model = load_model('saved_models/breast_cancer_lstm_model.keras')

def process_image(image):
    results = yolo_model(image, stream=True)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls)
                if class_id == 0:  
                    st.write("Tumor found in the image!")
        result_image = result.plot()
        return result_image

def preprocess_data(data):
    if "id" in data.columns:
        data = data.drop(columns=["id"])
    data = data.select_dtypes(include=[np.number])
    return data

def predict_diagnosis_for_dataset(model, input_data):
    predictions = model.predict(input_data)
    diagnosis = ["B" if pred[0] > pred[1] else "M" for pred in predictions]
    return diagnosis

# Streamlit application layout
st.title("Medical Diagnosis Application")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Brain Tumor Detection", "Breast Cancer Diagnosis", "CSV Format Guide"])

# Tab 1: Brain Tumor Detection
with tab1:
    st.header("Brain Tumor Detection using YOLOv8")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result_image = process_image(image_cv)
            st.image(result_image, caption='Result Image with Detections', use_column_width=True)

# Tab 2: Breast Cancer Diagnosis
with tab2:
    st.header("Breast Cancer Diagnosis Prediction")
    uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        original_ids = data["id"] if "id" in data.columns else None
        processed_data = preprocess_data(data)
        processed_data = processed_data.values.reshape(processed_data.shape[0], processed_data.shape[1], 1)
        predictions = predict_diagnosis_for_dataset(lstm_model, processed_data)
        results = pd.DataFrame()
        if original_ids is not None:
            results["id"] = original_ids
        results["predicted_diagnosis"] = predictions
        st.write("Predictions:")
        st.dataframe(results)
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

# Tab 3: CSV Format Guide
# Tab 3: CSV Format Guide
with tab3:
    st.header("CSV File Format Guide")
    st.write("To use the Breast Cancer Diagnosis Prediction feature, your CSV file should include the following columns in this specific order:")
    st.write("""
    - **id**: Unique identifier for each sample (optional).
    - **radius_mean**: Mean radius of the tumor.
    - **texture_mean**: Mean texture of the tumor.
    - **perimeter_mean**: Mean perimeter of the tumor.
    - **area_mean**: Mean area of the tumor.
    - **smoothness_mean**: Mean smoothness of the tumor.
    - **compactness_mean**: Mean compactness of the tumor.
    - **concavity_mean**: Mean concavity of the tumor.
    - **concave points_mean**: Mean number of concave points of the tumor.
    - **symmetry_mean**: Mean symmetry of the tumor.
    - **fractal_dimension_mean**: Mean fractal dimension of the tumor.
    - **radius_se**: Standard error of the radius.
    - **texture_se**: Standard error of the texture.
    - **perimeter_se**: Standard error of the perimeter.
    - **area_se**: Standard error of the area.
    - **smoothness_se**: Standard error of the smoothness.
    - **compactness_se**: Standard error of the compactness.
    - **concavity_se**: Standard error of the concavity.
    - **concave points_se**: Standard error of the number of concave points.
    - **symmetry_se**: Standard error of the symmetry.
    - **fractal_dimension_se**: Standard error of the fractal dimension.
    - **radius_worst**: Worst radius of the tumor.
    - **texture_worst**: Worst texture of the tumor.
    - **perimeter_worst**: Worst perimeter of the tumor.
    - **area_worst**: Worst area of the tumor.
    - **smoothness_worst**: Worst smoothness of the tumor.
    - **compactness_worst**: Worst compactness of the tumor.
    - **concavity_worst**: Worst concavity of the tumor.
    - **concave points_worst**: Worst number of concave points of the tumor.
    - **symmetry_worst**: Worst symmetry of the tumor.
    - **fractal_dimension_worst**: Worst fractal dimension of the tumor.
    """)
    
    st.write("Ensure there are no missing values in the numeric columns.")
    st.write("Example of a valid CSV format:")
    st.write("```\n"
             "id,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst\n"
             "1,12.34,15.67,80.45,500.25,0.1,0.2,0.3,0.1,0.5,0.2,1.2,1.3,2.1,5.5,0.05,0.08,0.12,0.07,0.15,0.04,14.56,18.90,90.12,600.60,0.15,0.20,0.25,0.14,0.25,0.08\n"
             "2,10.12,20.11,70.75,400.45,0.12,0.25,0.15,0.05,0.6,0.15,1.0,1.1,1.8,4.5,0.06,0.09,0.11,0.05,0.10,0.03,12.34,19.20,85.10,550.10,0.18,0.22,0.20,0.12,0.22,0.05\n"
             "```\n")
