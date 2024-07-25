import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Define categories
categories = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load and preprocess an uploaded image
def load_and_preprocess_image(image_bytes, img_size=(105, 105)):
    # Decode the image
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # Resize the image
    img_resized = cv2.resize(img, img_size)
    # Normalize the image
    img_normalized = img_resized / 255.0
    return img_normalized

# Load the trained model
def load_trained_model():
    model = load_model('cnn_model.h5')
    return model

# Predict the class of an image
def predict_image(model, image):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return categories[predicted_class]

# Streamlit app
def main():
    st.title("Image Classification with CNN")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        # Read the uploaded file as bytes
        image_bytes = uploaded_file.read()
        
        # Preprocess the image
        image = load_and_preprocess_image(image_bytes)
        
        # Load the model
        model = load_trained_model()
        
        # Predict the class
        prediction = predict_image(model, image)
        
        # Display the image
        st.image(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR), caption='Uploaded Image', use_column_width=True)
        
        # Display the prediction
        st.write(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
