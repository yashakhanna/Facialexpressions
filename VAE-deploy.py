import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained decoder model
decoder_path = 'C:/Users/dhill/VS-Code/deep-models-finetuned/vae_decoder.h5'
decoder = load_model(decoder_path, compile=False)

# Streamlit app
st.title('VAE Image Generator')

# Generate new images
st.write("Generate New Images")
num_images = st.slider('Number of images to generate', 1, 20, 10)

if st.button('Generate'):
    latent_dim = 100  # Make sure this matches the latent_dim used during training
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)

    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    for i in range(num_images):
        axes[i].imshow((generated_images[i] * 127.5 + 127.5).astype(np.uint8))
        axes[i].axis('off')
    st.pyplot(fig)
