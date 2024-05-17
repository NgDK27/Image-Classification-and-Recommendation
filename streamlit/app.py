import streamlit as st
import pandas as pd
from PIL import Image
from utils import load_model, prepare_image, predict, load_classes, load_styles, preprocess_and_predict, find_similar_items, recommendations_df, feature_extraction_model, load_cluster_model

st.title("Machine Learning (COSC2753): Group Assignment")

model1 = load_model('../data/models/task-1-CNN-best-backup.keras')
classes = load_classes('../data/label_encoders/class_encoder.npy')

# feature_extraction_model = load_model('../data/models/fe-cnn')
cluster_model = load_cluster_model('../data/models/cluster-kmeans.model')
# # recommendations_df = pd.read_csv('../path/to/recommendations.csv')
# # database_features = recommendations_df.filter(regex='^x[0-9]+').values

model3 = load_model('../data/models/task-3-CNN.keras')
styles = load_styles('../data/label_encoders/style_encoder.npy')



st.header("Upload the furniture image: ")
uploaded_file = st.file_uploader("'.png' and 'jpg' only: ", type=["png", "jpg"], key="1")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=256)


st.header("Model 1: Classify furniture kind")
if uploaded_file is not None:
    image_array = prepare_image(uploaded_file)
    if image_array is not None:
        predictions1, predicted1_index = predict(model1, image_array)
        if predictions1 is not None:
            # Display predictions with labels
            st.write('< Same Furniture Class Prediction >')
            for label, pred in zip(classes, predictions1[0]):
                st.write(f"- {label}: {pred:.8f}")
            # Highlight the most probable label
            most_likely_class = classes[predicted1_index]
            most_likely_prob = predictions1[0][predicted1_index]
            st.write('< Same Furniture Class Result >')
            st.write(f"- {most_likely_class} ({most_likely_prob:.8f})")
    else:
        st.error("Failed to process the image. Please check the logs for more information.")


st.header("Model 2: Show different & visually similar 10 images of the same kind")
if uploaded_file is not None:
    if predictions1 is not None:
        # preprocessing, feature vector extraction, find similar kind
        feature_vector = preprocess_and_predict(image_array)
        top_indices = find_similar_items(feature_vector)

        # display
        for idx in top_indices:
            item_path = recommendations_df.iloc[idx]['Path']
            item_image = Image.open('../data/raw/Furniture_Data/' + item_path)
            st.image(item_image, caption=f"Similar Furniture: {recommendations_df.iloc[idx]['Style']} {recommendations_df.iloc[idx]['Class']}")
        

st.header("Model 3: Classify furniture style")
if uploaded_file is not None:
    image_array = prepare_image(uploaded_file)
    if image_array is not None:
        predictions3, predicted3_index = predict(model3, image_array)
        if predictions3 is not None:
            # Display predictions with labels
            st.write('< Same Interior Style Prediction >')
            for label, pred in zip(styles, predictions3[0]):
                st.write(f"- {label}: {pred:.8f}")
            # Highlight the most probable label
            most_likely_style = styles[predicted3_index]
            most_likely_prob = predictions3[0][predicted3_index]
            st.write('< Same Interior Style Result >')
            st.write(f"- {most_likely_style} ({most_likely_prob:.8f})")
        else:
            st.error("Failed to make prediction. Please check the logs for more information.")
    else:
        st.error("Failed to process the image. Please check the logs for more information.")