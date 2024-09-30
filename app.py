import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import easyocr
import tensorflow as tf
from groq import Groq

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

def extract_text_from_image(image):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image, detail=0)
    return ' '.join(result)

def get_bot_response(user_input, extracted_text):
    try:
        client = Groq(api_key="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")
        prompt = f"Context: {extracted_text}\nUser Question: {user_input}"
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            model="llama3-8b-8192",  
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None


def preprocess_image(img, img_size=(96, 96)):
    if img is not None:
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        return img
    else:
        st.error("Image could not be loaded.")
        return None

def predict_image(model_path, img):
    model = tf.keras.models.load_model(model_path)
    preprocessed_img = preprocess_image(img)
    if preprocessed_img is not None:
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return predicted_class
    else:
        return None



st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        font-family: 'Arial';
        color: #4CAF50;
        text-align: center;
    }
    .subheader {
        font-size: 30px;
        color: #333;
        text-align: left;
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 50px;
    }
    .container {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stTabs {
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='title'>💻 Cancer Diagnosis & Detection System </h1>", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='text-align: center;'>Navigation</h3>", unsafe_allow_html=True)
tabs = ["🧠 Brain Tumor Detection", "👩‍⚕️ Breast Cancer Diagnosis", "💊 Prescription Q&A", "📊 CSV Format Guide", "🔬 Pathological Tumor Detection"]
selection = st.sidebar.radio("Go to", tabs)

# Tab 1: Brain Tumor Detection using YOLO
if selection == "🧠 Brain Tumor Detection":
    st.markdown("<h2 class='subheader'>🧠 Brain Tumor Detection using YOLOv8</h2>", unsafe_allow_html=True)
    
    with st.expander("📥 Upload Brain Images", expanded=True):
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        cols = st.columns(len(uploaded_files))
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            cols[idx].image(image, caption=f'Uploaded Image {idx+1}', use_column_width=True)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner('🔍 Processing image...'):
                result_image = process_image(image_cv)

            st.image(result_image, caption=f'Result Image with Detections {idx+1}', use_column_width=True)

# Tab 2: Breast Cancer Diagnosis
elif selection == "👩‍⚕️ Breast Cancer Diagnosis":
    st.markdown("<h2 class='subheader'>👩‍⚕️ Breast Cancer Diagnosis Prediction</h2>", unsafe_allow_html=True)
    
    with st.expander("📥 Upload CSV File"):
        uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        original_ids = data["id"] if "id" in data.columns else None
        processed_data = preprocess_data(data)
        processed_data = processed_data.values.reshape(processed_data.shape[0], processed_data.shape[1], 1)

        with st.spinner('🔍 Predicting...'):
            predictions = predict_diagnosis_for_dataset(lstm_model, processed_data)

        results = pd.DataFrame()
        if original_ids is not None:
            results["id"] = original_ids
        results["predicted_diagnosis"] = predictions

        st.success("✅ Predictions generated successfully!")
        st.dataframe(results)
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

# Tab 3: Prescription Q&A
elif selection == "💊 Prescription Q&A":
    st.markdown("<h2 class='subheader'>💊 Prescription Question-Answering System</h2>", unsafe_allow_html=True)
    
    subtab1, subtab2 = st.tabs(["📥 Upload Prescription", "❓ Ask Questions"])

    with subtab1:
        st.write("Upload your prescription, and we'll extract the text for you.")
        uploaded_file_prescription = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file_prescription is not None:
            image_prescription = Image.open(uploaded_file_prescription)
            st.image(image_prescription, caption='Uploaded Prescription', use_column_width=True)
            opencv_image = np.array(image_prescription)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

            with st.spinner('🔍 Extracting text from the image...'):
                extracted_text = extract_text_from_image(opencv_image)

            if extracted_text:
                st.subheader("Extracted Text:")
                st.success(extracted_text)
            else:
                st.warning("No text could be extracted from the image. Please try again.")

    # Subtab 2: Ask Questions from Prescription
    with subtab2:
        st.write("Ask questions about the prescription.")
        
        if uploaded_file_prescription is None or not extracted_text:
            st.warning("⚠️ Please upload a prescription and extract text from it in the 'Upload Prescription' tab first.")
        else:
            user_input = st.text_input("Ask a question about the prescription:")
            submit_button = st.button("Submit")

            if submit_button and user_input:
                st.spinner("🔍 Generating response...") 
                response = get_bot_response(user_input, extracted_text)
                if response:
                    st.success(f"🗣️ Response: {response}")
                else:
                    st.error("⚠️ No response received. Check the input or try again.")

# Tab 4: CSV Format Guide
elif selection == "📊 CSV Format Guide":
    st.markdown("<h2 class='subheader'>📊 CSV File Format Guide</h2>", unsafe_allow_html=True)
    
    st.write("To use the Breast Cancer Diagnosis Prediction feature, your CSV file should include the following columns in this specific order:")
    st.markdown("""
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
    
    st.info("If you upload a CSV file without the **id** column, the system will automatically generate predictions based on the remaining columns.")

# Tab 5: Pathological Tumor Detection
elif selection == "🔬 Pathological Tumor Detection":
    st.markdown("<h2 class='subheader'>🔬 Pathological Tumor Detection</h2>", unsafe_allow_html=True)
    
    with st.expander("📥 Upload Tumor Images"):
        uploaded_file_tumor = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png","tiff","tif"])

    if uploaded_file_tumor is not None:
        image = Image.open(uploaded_file_tumor)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        with st.spinner('🔍 Predicting tumor...'):
            predicted_class = predict_image('saved_models/model_pathological_images.keras', opencv_image)

        if predicted_class == 1:
            st.success("🟢 Tumor detected!")
        elif predicted_class == 0:
            st.success("🔵 Tumor not detected.")
        else:
            st.warning("⚠️ Prediction could not be made.")