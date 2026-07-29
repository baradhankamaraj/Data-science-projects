import streamlit as st
import numpy as np
import tensorflow as tf

# Load model
best_model = tf.keras.models.load_model("best_currency_model.keras")

# Class labels
class_names = ['10', '100', '20', '200', '2000', '50', '500', 'Invalid']

st.title("Indian Currency Detection")

uploaded_file = st.file_uploader(
    "Upload a Currency Image",
    type=["jpg", "jpeg", "png"]
    
)

if uploaded_file is not None:

    # Display image
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")

    # Load image
    img = tf.keras.utils.load_img(
    uploaded_file,
    target_size=(128, 128),
    color_mode="rgb",
    interpolation="bilinear"
)

    # Convert to array
    img_array = tf.keras.utils.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)

    # Prediction
    prediction = best_model.predict(img_array,verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Currency: ₹{class_names[predicted_class]}")

    st.write(f"Confidence: {confidence:.2f}%")