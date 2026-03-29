import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(page_title="Plant Disease Predictor", page_icon="🌿")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenet_crop_disease.keras')

model = load_model()

class_names = [
    'Pepper__bell___Bacterial_spot', 
    'Potato___healthy', 
    'Tomato_Leaf_Mold', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato_Bacterial_spot', 
    'Tomato_Septoria_leaf_spot', 
    'Tomato_healthy', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato_Early_blight', 
    'Tomato__Target_Spot', 
    'Pepper__bell___healthy', 
    'Potato___Late_blight', 
    'Tomato_Late_blight', 
    'Potato___Early_blight', 
    'Tomato__Tomato_mosaic_virus'
]


st.title("🌿 Crop Disease AI Diagnostician")
st.write("Upload a clear picture of a plant leaf, and our custom MobileNetV2 brain will identify the disease and confidence level.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    if st.button('Analyze Leaf'):
        with st.spinner('The AI is looking at the visual features...'):
            
            img = image.resize((224, 224))
            
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            predicted_class = class_names[predicted_index]
            
            clean_name = predicted_class.replace('___', ' - ').replace('__', ' ').replace('_', ' ')

            st.success("Analysis Complete!")
            st.subheader(f"Diagnosis: **{clean_name}**")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            if confidence > 90:
                st.balloons()