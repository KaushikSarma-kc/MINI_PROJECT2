
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = load_model('/content/drive/MyDrive/potato_model.h5')

# Define the prediction function
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence

# Define the Streamlit app
def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image to know if it has early blight or late blight")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Save the uploaded image to a temporary location
        temp_img_path = "temp_image.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict the class and confidence
        predicted_class, confidence = predict_image(temp_img_path, model)

        # Display the prediction
        if predicted_class == 0:
            st.write(f"The image is predicted to have Early Blight with {confidence*100:.2f}% confidence.")
        elif predicted_class == 1:
            st.write(f"The image is predicted to have Late Blight with {confidence*100:.2f}% confidence.")
        else:
            st.write("Unknown class")

if __name__ == "__main__":
    main()
