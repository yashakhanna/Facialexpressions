import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
import os
import cv2

# Function to load the trained generator model
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_generator():
    return load_model('C:/Users/dhill/VS-Code/generator_model_final.h5')

generator = load_generator()

# Define categories and latent dimension
categories = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
LATENT_DIM = 100
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

st.title("cGAN Image Generator")

st.write("This app generates images based on different expressions using a conditional GAN.")

# Display sample images (assuming they are already generated and saved)
for category in categories:
    st.header(f"Generated images for {category}")
    img_path = f"{output_dir}/{category}_0.png"
    if os.path.exists(img_path):
        st.image(img_path, caption=f"{category} - Sample 1")
    else:
        st.warning(f"Image not found: {img_path}")

# Option to generate new images
if st.button('Generate New Images'):
    st.write("Generating new images...")

    # Generate new images
    random_latent_vectors = np.random.normal(size=(len(categories), LATENT_DIM))
    generated_images = generator.predict(random_latent_vectors)
    generated_images = 0.5 * generated_images + 0.5  # Rescale images 0-1

    # Save and display generated images
    for i, category in enumerate(categories):
        img = generated_images[i]
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        # Save the image
        img_path = f"{output_dir}/{category}_0.png"
        cv2.imwrite(img_path, img)

        # Display the image
        st.image(img, caption=f"New {category} image")

        # Debugging statement to verify the path and image existence
        st.write(f"Image saved at: {img_path}")
        if os.path.exists(img_path):
            st.write(f"Image exists at: {img_path}")
        else:
            st.write(f"Failed to save image at: {img_path}")

# Note: Replace 'generator_model_final.h5' with the path to your trained generator model
