import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (400,400)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        
        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('bestmodel.h5')

st.write("""
         # Diabetic-Retinopathy Classifier
         """
         )

st.write("This is a simple image classification web app to predict DR severity")

P_name = st.text_input("Enter Full Name")

P_no = st.text_input("Enter Mobile Number")

P_ID = st.text_input("Enter Unique ID")

file = st.file_uploader("Please upload an image file", type=["jpg", "png","tif"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("ZERO :  No DR")
    elif np.argmax(prediction) == 1:
        st.write("One : Mild")
    elif np.argmax(prediction) == 2:
        st.write("Two : Moderate")
    elif np.argmax(prediction) == 3:
        st.write("Three : Severe")
    else:
        st.write("Four : Proliferative DR")
    
    st.text("Probability (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR)")
    st.write(prediction)