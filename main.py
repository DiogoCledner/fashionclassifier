import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/modelo/fashion_mnist_model_treinado.h5"
new_model_path = f"{working_dir}/modelo/fashion_mnist_model_resaved.h5"

# Load the pre-trained model without compiling
model = tf.keras.models.load_model(model_path, compile=False)

# Re-save the model to the new path
model.save(new_model_path)

# Define custom objects if needed (empty in this case)
custom_objects = {}

# Load the re-saved model with custom objects
model = tf.keras.models.load_model(new_model_path, custom_objects=custom_objects)

# Define class labels for Fashion MNIST dataset
class_names = ['Camiseta/Top', 'Calça', 'Pullover', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Sapatinha', 'Bolsa', 'Botas']

# Streamlit App
st.title('Classificador de itens de moda')

uploaded_image = st.file_uploader("Carregar uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classificar'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Predição: {prediction}')
