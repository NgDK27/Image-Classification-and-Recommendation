import logging
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
# from sklearn.cluster import KMeans
import os
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model



logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


### Common Functions
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
    if model is not None and image_array is not None:
        try:
            predictions = model.predict(image_array)
            logging.info("Prediction executed successfully.")
            predicted_index = np.argmax(predictions, axis=1)[0]
            return predictions, predicted_index
        except Exception as e:
            logging.error("Failed to make predictions.", exc_info=True)
            return None
    else:
        logging.error("Model or image array is None, prediction cannot be performed.")
        return None



### Task-1-CNN
def load_classes(filename):
    class_data = np.load(filename, allow_pickle=True)
    return class_data.tolist()



### Task-2
feature_extraction_model = load_model('../data/models/fe-cnn')
recommendations_df = pd.read_csv('../data/recommend/csv/recommendations.csv')

database_features = recommendations_df.filter(regex='^x[0-9]+').values

@st.cache_resource
def preprocess_and_predict(image_array):
    feature_vector = feature_extraction_model.predict(image_array)
    return feature_vector

@st.cache_resource
def find_similar_items(feature_vector):
    similarities = cosine_similarity(feature_vector, database_features)
    top_indices = np.argsort(similarities[0])[::-1][:10]
    return top_indices

def load_cluster_model(model_path):
    cluster_model = load(model_path)
    return cluster_model


### Task-3-CNN
def load_styles(filename):
    style_data = np.load(filename, allow_pickle=True)
    return style_data.tolist()[0]