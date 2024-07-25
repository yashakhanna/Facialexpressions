import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import streamlit as st
import io

# Function to load and preprocess the model
def load_model():
    model_path = "C:/Users/hidde/OneDrive/Documents/FaceExpressions/VGG_finetuned.keras"  
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess the image
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data))
    img = img.resize(target_size)
    img = np.asarray(img) / 255.0  # Normalize pixel values to [0, 1]
    img = img.astype(np.float32)  
    img = np.expand_dims(img, axis=0)  
    return img

# Streamlit app
def main():
    st.set_page_config(page_title="Facial Expression Classification", layout="wide")
    
    st.markdown("<h1 style='text-align: center; color: white; background-color: #4CAF50; padding: 10px;'>Facial Expression Classification</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;'>Upload an image to classify</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type="jpg")

    if uploaded_file is not None:
        try:
            model = load_model()
            image = preprocess_image(uploaded_file)
            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            class_names = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
            confidence = prediction[0][class_index] * 100

            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
            with col2:
                st.markdown(f"<h3>Predicted Expression: <span style='color: #4CAF50;'>{class_names[class_index]}</span></h3>", unsafe_allow_html=True)
                st.markdown(f"<h4>Confidence: <span style='color: #4CAF50;'>{confidence:.2f}%</span></h4>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                for i, class_name in enumerate(class_names):
                    st.markdown(f"<p>{class_name}: {prediction[0][i] * 100:.2f}%</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown("<footer style='text-align: center; padding: 10px; background-color: #4CAF50; color: white;'>Developed by [Prabhot SIngh]</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
