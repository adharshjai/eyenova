import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np


kfr_model = tf.keras.models.load_model('kfr_model.h5')
glaucoma_model = tf.keras.models.load_model('glaucoma_model.h5')


def predict_kfr(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = kfr_model.predict(img_array)
    return prediction[0][0]


def predict_glaucoma(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = glaucoma_model.predict(img_array)
    return prediction[0][0]


st.title('EyeNova: Kayser-Fleischer Ring and Glaucoma Detection')
st.write('Upload an image for detection.')

uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.write('Analyzing...')

    # Kayser-Fleischer Ring Prediction
    kfr_prediction = predict_kfr(img)

    # Glaucoma Prediction
    glaucoma_prediction = predict_glaucoma(img)

    if kfr_prediction > 0.5 and glaucoma_prediction > 0.5:
        if kfr_prediction > glaucoma_prediction:
            st.warning('Kayser-Fleischer rings detected!')
        else:
            st.warning('Glaucoma Positive!')
    elif kfr_prediction > 0.5:
        st.warning('Kayser-Fleischer rings detected!')
    elif glaucoma_prediction > 0.5:
        st.warning('Glaucoma Positive!')
    else:
        st.success('No Kayser-Fleischer rings or Glaucoma detcted.')
