import logging
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import os
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model


# Configure logging to help in debugging and log management
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


### Common Functions
@st.cache_resource  # Use Streamlit's caching to avoid reloading model from disk repeatedly
def load_model(model_path):
    if os.path.exists(model_path):  # Check if the model path exists
        try:
            model = tf.keras.models.load_model(model_path)  # Load the Keras model
            logging.info(f"Model loaded successfully from {model_path}.")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}.", exc_info=True)
            return None
    else:
        logging.warning(f"Model path {model_path} does not exist. Please check the path.")
        return None


### Task-1-CNN
def load_classes(filename):
    class_data = np.load(filename, allow_pickle=True)  # Load class data using numpy
    return class_data.tolist()

def predict(model, image_array):
    if model is not None and image_array is not None:  # Ensure model and image array are not None
        try:
            predictions = model.predict(image_array)  # Make predictions using the model
            logging.info("Prediction executed successfully.")
            predicted_index = np.argmax(predictions, axis=1)[0]  # Find the index of the highest probability
            return predictions, predicted_index
        except Exception as e:
            logging.error("Failed to make predictions.", exc_info=True)
            return None
    else:
        logging.error("Model or image array is None, prediction cannot be performed.")
        return None

def prepare_image(uploaded_file):
    try:
        # Process the image for model input
        image = Image.open(uploaded_file).convert('RGBA')  # Convert image to RGBA to handle transparency
        background = Image.new('RGBA', image.size, (255, 255, 255))  # Create a white background
        image_with_background = Image.alpha_composite(background, image).convert('RGB')  # Composite with white background

        image_resized = image_with_background.resize((256, 256))  # Resize image to expected model input size
        image_array = np.array(image_resized) / 255.0  # Convert to array and normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension for model input

        logging.info("Image processed successfully.")
        return image_array
    except Exception as e:
        logging.error("Failed to process the image: " + str(e), exc_info=True)
        st.error("Failed to process the image. Please try another file.")  # Show error in Streamlit
        return None


### Task-2
feature_extraction_model = load_model('../data/models/fe-cnn')  # Load the feature extraction model
recommendations_df = pd.read_csv('../data/recommend/csv/recommendations.csv')  # Load recommendations data

@st.cache_resource
def preprocess_and_predict(image_array):
    feature_vector = feature_extraction_model.predict(image_array)  # Extract features from the image
    return feature_vector

@st.cache_resource
def find_similar_items(feature_vector, filtered_recommendations):
    database_features = filtered_recommendations.filter(regex='^x[0-9]+').values  # Extract feature columns
    cosine_similarities = cosine_similarity(feature_vector, database_features)  # Compute cosine similarity
    top_indices = np.argsort(cosine_similarities[0])[::-1][:10]  # Get top 10 similar items
    return top_indices


### Task-3-CNN
def load_styles(filename):
    style_data = np.load(filename, allow_pickle=True)  # Load style data using numpy
    return style_data.tolist()[0]

def load_cluster_model(model_path):
    cluster_model = load(model_path)  # Load a clustering model
    return cluster_model

def classify_and_recommend(image_array, class_model, style_model, threshold=0.3, close_threshold=0.01):
    # Load the labels for classes and styles from predefined encoders
    classes = load_classes('../data/label_encoders/class_encoder.npy')
    styles = load_styles('../data/label_encoders/style_encoder.npy')
    
    # Predict the class and style from the provided models
    class_pred = class_model.predict(image_array)
    style_pred = style_model.predict(image_array)

    # Combine style labels with their corresponding prediction probabilities
    all_predictions = list(zip(styles, style_pred.flatten().tolist()))

    # Identify the most likely class from the predictions
    predicted_class_idx = np.argmax(class_pred, axis=1)[0]
    predicted_class = classes[predicted_class_idx]

    # Use close_threshold to select multiple closely scored styles
    max_style_prob = np.max(style_pred)
    predicted_style_indices = [i for i, prob in enumerate(style_pred.flatten()) if prob >= max_style_prob - close_threshold]
    predicted_styles = [styles[i] for i in predicted_style_indices]

    # Filter recommendations based on the predicted class and selected styles
    filtered_recommendations = recommendations_df[
        (recommendations_df['Class'] == predicted_class) & 
        (recommendations_df['Style'].isin(predicted_styles))
    ]

    # Extract the feature vector of the image using the feature extraction model
    image_feature_vector = feature_extraction_model.predict(image_array).reshape(1, -1)
    # Retrieve feature data from the filtered recommendations
    filtered_features = filtered_recommendations.filter(regex='^x[0-9]+').values
    
    cosine_similarities = cosine_similarity(image_feature_vector, filtered_features)  # Compute cosine similarities
    top_indices = np.argsort(-cosine_similarities.flatten())[:10]  # Top 10 recommendations
    top_recommendations = filtered_recommendations.iloc[top_indices]

    return predicted_class, predicted_styles, all_predictions, top_recommendations