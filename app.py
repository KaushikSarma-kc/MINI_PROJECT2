import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to load the model and make predictions
@st.cache_data
def load_model():
    model = tf.keras.models.load_model('K:/MINI_PROJ_2/Deep_Learning_POTATO/potato_model.h5')  
    return model

# Main function to run the Streamlit app
def main():
    st.title('Potato Image Classifier')
    st.write('Upload an image of a potato to classify its type.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        # Display "Classifying..." while processing
        classification_text = st.empty()
        classification_text.write("Classifying...")

        model = load_model()
        image_array = load_and_preprocess_image(uploaded_file)
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)

        # Define your label names corresponding to each class index
        label_names = {
            0: 'Early Blight',
            1: 'Late Blight',
            2: 'Healthy Potato'
        }

        # Get the predicted label name
        predicted_label = label_names[predicted_class]

        # Remove the "Classifying..." text
        classification_text.empty()

        # Display prediction with larger font size and different color
        st.markdown(f"<h2 style='color: green; font-size: 36px;'>Prediction: {predicted_label}</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
