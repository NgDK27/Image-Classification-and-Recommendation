import logging
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import random
import tensorflow as tf
import os


logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Model loaded successfully from {model_path}.")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}.", exc_info=True)
            return None
    else:
        logging.warning(f"Model path {model_path} does not exist. Please check the path.")
        return None


def prepare_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((256, 256))  # resizing
        image_array = np.array(image) / 255.0  # normalization
        image_array = np.expand_dims(image_array, axis=0)  # batch processing
        logging.info("Image processed successfully.")
        return image_array
    except Exception as e:
        logging.error("Failed to process the image: " + str(e), exc_info=True)
        st.error("Failed to process the image. Please try another file.")
        return None


def predict(model, image_array):
    class_labels_dict = {
        0: 'beds',
        1: 'chairs',
        2: 'sofas',
        3: 'lamps',
        4: 'tables',
        5: 'dressers'
    }

    if model is not None and image_array is not None:
        try:
            predictions = model.predict(image_array)
            logging.info("Prediction executed successfully.")
            # Find the index of the class with the highest probability and convert it to the corresponding class label
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = class_labels_dict[predicted_index]
            return predictions, predicted_label
        except Exception as e:
            logging.error("Failed to make predictions.", exc_info=True)
            return None
    else:
        logging.error("Model or image array is None, prediction cannot be performed.")
        return None


st.title("Machine Learning (COSC2753): Group Assignment")

tab1, tab2, tab3 = st.tabs(["Task 1", "Task 2", "Task 3"])

model1 = load_model('data/models/task-1-CNN.keras')
# model2 = load_model('data/models/task-2.keras')
# model3 = load_model('data/models/task-3.keras')


with tab1:
    st.header("Model 1: Classify furniture kind")
    uploaded_file_1 = st.file_uploader("Upload the furniture image: ", type=["png", "jpg"], key="1")

    if uploaded_file_1 is not None:
        image1_array = prepare_image(uploaded_file_1)
        if image1_array is not None:
            predictions1, predicted_label = predict(model1, image1_array)
            if predictions1 is not None:
                image_1 = Image.open(uploaded_file_1)
                st.image(image_1, caption='Uploaded Image', width=256)
                st.write(f'Class Labels: [beds, chairs, sofas, lamps, tables, dressers]')
                st.write(f'Prediction: {predictions1}')
                st.write(f'Classification Result: {predicted_label}')
            else:
                st.error("Failed to make prediction. Please check the logs for more information.")
        else:
            st.error("Failed to process the image. Please check the logs for more information.")


with tab2:
    st.header("Model 2: Show different & visually similar 10 images of the same kind")
    uploaded_file_2 = st.file_uploader("Upload the furniture image: ", type=["png", "jpg"], key="2")


with tab3:
    st.header("Model 3: Classify furniture style")
    uploaded_file_3 = st.file_uploader("Upload the furniture image: ", type=["png", "jpg"], key="3")

