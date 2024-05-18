import streamlit as st
from PIL import Image
from utils import (load_model, prepare_image, predict, load_classes, load_styles, 
                   preprocess_and_predict, find_similar_items, recommendations_df, 
                   classify_and_recommend, load_cluster_model)


st.title("Machine Learning (COSC2753): Group Assignment")

# Load the models and label encoders
model1 = load_model('../data/models/task-1-CNN.keras')
classes = load_classes('../data/label_encoders/class_encoder.npy')
model3 = load_model('../data/models/task-3-CNN.keras')
styles = load_styles('../data/label_encoders/style_encoder.npy')
cluster_model = load_cluster_model('../data/models/cluster-kmeans.model')


# Interface for uploading an image
st.markdown("### Upload the furniture image: ")
uploaded_file = st.file_uploader("Please upload an image file: ", type=["png", "jpg", "webp"], key="1")  # Create an image file uploader


# Furniture Kind classification section
st.markdown("### Task 1: Classify furniture class")
if uploaded_file is not None:  # Check if a file has been uploaded
    image = Image.open(uploaded_file)  # Open the uploaded file as an image
    st.image(image, caption='Uploaded Image', width=256)  # Display the uploaded image
    image_array = prepare_image(uploaded_file)  # Preprocess the image for model input
    if image_array is not None:
        predictions1, predicted1_index = predict(model1, image_array)  # Predict the class of the furniture using the task1 model
        if predictions1 is not None:
            st.markdown('##### ① Same Furniture Class Prediction:')
            for label, pred in zip(classes, predictions1[0]):  # Loop through class labels and predictions
                st.write(f"- {label}: {pred:.8f}")  # Display each class label and its prediction probability
            most_likely_class = classes[predicted1_index]  # Determine the class with the highest probability
            most_likely_prob = predictions1[0][predicted1_index]  # Get the highest probability value
            st.markdown('##### ② Same Furniture Class Result:')  # Display the most likely class and its probability
            st.write(f"- {most_likely_class} ({most_likely_prob:.8f})")
    else:
        st.error("Failed to process the image. Please check the logs for more information.")  # Display an error if image processing fails


# Recommending visually similar furniture images
st.markdown("### Task 2: Recommend visually similar 10 furnitures")
if uploaded_file is not None and most_likely_class:
    image = Image.open(uploaded_file)  # Reopen the uploaded file for display
    st.image(image, caption='Uploaded Image', width=256)  # Display the image again
    feature_vector = preprocess_and_predict(image_array)  # Extract feature vectors from the image
    filtered_recommendations = recommendations_df[recommendations_df['Class'] == most_likely_class]  # Filter recommendations based on the predicted class
    top_indices = find_similar_items(feature_vector, filtered_recommendations)  # Find top visually similar items
    st.markdown('##### ① Visually Similar Furniture:')
    cols = st.columns(3)  # Create a layout with 3 columns for the visualization
    count = 0
    for idx in top_indices:
        item_path = filtered_recommendations.iloc[idx]['Path']  # Get the path of each recommended item
        item_image = Image.open('../data/raw/Furniture_Data/' + item_path)  # Open the image from the path
        with cols[count % 3]:
            st.image(item_image, caption=f"{filtered_recommendations.iloc[idx]['Style']} {filtered_recommendations.iloc[idx]['Class']}", width=150)  # Display each image in a column with a caption indicating its style and class
        count += 1


# Recommending not only the same class but also the same style of furnitures
st.markdown("### Task 3: Classify furniture style & Recommend the same style furniture")
if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Reopen the uploaded file for display
    st.image(image, caption='Uploaded Image', width=256)  # Display the image again
    if image_array is not None:
        predictions3, predicted3_index = predict(model3, image_array)  # Use the task3 model to predict the style of the furniture
        if predictions3 is not None:
            st.markdown('##### ①  Same Interior Style Prediction:')
            for label, pred in zip(styles, predictions3[0]):  # Loop through style labels and predictions
                st.write(f"- {label}: {pred:.8f}")  # Display each style label and its prediction probability
            predicted_class, predicted_styles, recommendations = classify_and_recommend(image_array, model1, model3)  # Classify and recommend based on the image
            most_likely_style = styles[predicted3_index]  # Determine the style with the highest probability
            most_likely_prob = predictions3[0][predicted3_index]  # Get the highest probability value for the style
            st.markdown('##### ② Same Interior Style Result:')  # Display the most & multiple likely style
            for style in predicted_styles:
                st.write(f"- {style}")
            if not recommendations.empty:
                st.markdown('##### ③ Recommendations')  # Display recommendations if any are found
                cols = st.columns(3)  # Create a layout with 3 columns for recommendations
                count = 0
                for idx, row in recommendations.iterrows():
                    rec_image = Image.open(f'../data/raw/Furniture_Data/{row["Path"]}')  # Open each recommended image
                    with cols[count % 3]:
                        st.image(rec_image, caption=f"{row['Style']} {row['Class']}", width=150)  # Display each recommended image in a column with a caption
                    count += 1
            else:
                st.error("No recommendations found matching the criteria.")  # Display an error if no recommendations are found
    else:
        st.error("Failed to process the image. Please check the logs for more information.")  # Display an error if image processing fails