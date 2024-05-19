import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from utils import (load_model, prepare_image, predict, load_classes, load_styles, 
                   preprocess_and_predict, find_similar_items, recommendations_df, 
                   classify_and_recommend, load_cluster_model)

st.title("COSC2753: Group Project")
st.header("T3_Group 5")

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
            sizes = predictions1[0]  # prediction for each class
            # Plotly graph
            fig = go.Figure(data=[go.Pie(labels=classes, values=sizes, textinfo='label+percent', hole=.3)])
            # fig.update_traces(hoverinfo='label+percent', textfont_size=12, marker=dict(line=dict(color='#000000', width=1)))
            st.plotly_chart(fig, use_container_width=True)

            most_likely_class = classes[predicted1_index]  # Determine the class with the highest probability
            most_likely_prob = predictions1[0][predicted1_index]  # Get the highest probability value
            st.markdown('##### ② Same Furniture Class Result:')
            st.write(f"- {most_likely_class} ({most_likely_prob:.8f}%)")
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
        predicted_class, predicted_styles, all_predictions, recommendations = classify_and_recommend(image_array, model1, model3)
        if all_predictions is not None:
            st.markdown('##### ① Same Interior Style Predictions:')
            for style, probability in all_predictions:
                st.write(f"- {style}: {probability:.8f}")  # Display each style and its probability
            # Separate labels and probabilities
            labels = [pred[0] for pred in all_predictions]
            values = [pred[1] for pred in all_predictions]
            # Use Plotly Express's color cycle to assign colors to each label
            colors = px.colors.qualitative.Plotly  # Use Plotly's default color palette
            # Generate text for probability percentages
            text = [f"{value:.2%}" for value in values]
            fig = go.Figure(go.Bar(
                x=values,  # Probabilities on the x-axis
                y=labels,  # Labels on the y-axis
                orientation='h',  # Horizontal bar chart
                text=text,  # Text to display on each bar (percentages)
                textposition='auto',  # Set text position inside the bars
                marker_color=[colors[i % len(colors)] for i in range(len(labels))]  # Apply color cycle to each label
            ))
            # Style the graph
            fig.update_layout(
                xaxis_title='Probability',
                yaxis_title='Style',
                yaxis={'categoryorder':'total ascending'},  # Sort by probability
                xaxis=dict(range=[0, 1])  # Set x-axis range from 0 to 1
                )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('##### ② Furniture Style Classfication Result: ')
            if all_predictions:
                # Find the style with the highest probability
                most_likely_style, highest_probability = max(all_predictions, key=lambda x: x[1])
                # Flag to check if any style exceeds the threshold
                exceeded_threshold = False  
                # List to store styles that are close to the highest probability
                close_styles = []  
                # Loop through all predictions to display styles that exceed the threshold
                # Also, collect styles that are close to the highest probability within a 1% range
                for style, probability in all_predictions:
                    if probability > 0.3:  # Check if the probability exceeds the threshold
                        st.write(f"- {style} ({probability:.8f}%)")
                        exceeded_threshold = True
                    elif probability >= highest_probability - 0.01:  # Check if within 1% of the highest probability
                        close_styles.append((style, probability))
                # If no style exceeds the threshold, display styles close to the highest probability
                if not exceeded_threshold:
                    if close_styles:
                        for style, prob in close_styles:
                            st.write(f"- {style} ({prob:.8f}%)")
                    else:
                        # If there are no close styles, display only the style with the highest probability
                        st.write(f"- {most_likely_style} ({highest_probability:.8f}%)")

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