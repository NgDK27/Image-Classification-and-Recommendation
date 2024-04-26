import pandas as pd
from tqdm import tqdm
import os
import imagehash
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# Fonts for visualization
font = font_manager.FontProperties(family='sans-serif', weight='bold')
file = font_manager.findfont(font)


# Find duplicates and similar images, then output them in a DataFrame
def find_near_duplicates(df, threshold=5, base_path="../data/raw/Furniture_Data", ):
    image_hashes = {}
    duplicates = []
    total_images = len(df)

    # Create a tqdm progress bar
    progress_bar = tqdm(total=total_images, unit='image', desc='Processing images', leave=True)

    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        file_path = f"{base_path}/{row['Path']}"
        progress_bar.set_postfix({'Duplicates': len(duplicates), 'Current': file_path})

        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Calculate perceptual hash of the image
            image = Image.open(file_path)
            image_hash = imagehash.phash(image)

            # Check if a similar hash already exists
            is_duplicate = False
            for existing_hash, paths in image_hashes.items():
                if image_hash - existing_hash <= threshold:
                    paths.append(row)
                    duplicates.append(paths)
                    is_duplicate = True
                    break

            # If the image is not a duplicate, add its hash to the dictionary
            if not is_duplicate:
                image_hashes[image_hash] = [row]

        progress_bar.update(1)  # Update the progress bar
        progress_bar.set_postfix({'Duplicates': len(duplicates), 'Current': file_path})

    progress_bar.close()  # Close the progress bar

    # Create a new DataFrame to store the duplicates
    duplicates_df = pd.DataFrame(
        columns=['Duplicate_Group', 'First_Image_Path', 'Duplicate_or_Similar'] + list(df.columns))

    # Iterate over the duplicate groups
    for i, duplicate_group in enumerate(duplicates, start=1):
        if len(duplicate_group) > 1:
            first_image_path = duplicate_group[0]['Path']

            # Determine the values for the "Duplicate_or_Similar" column
            styles = [row['Style'] for row in duplicate_group]
            similar_or_duplicate = []
            for style in styles:
                if styles.count(style) > 1:
                    similar_or_duplicate.append('Duplicate')
                else:
                    similar_or_duplicate.append('Similar')

            # Add the duplicate rows to the new DataFrame
            for j, duplicate_row in enumerate(duplicate_group):
                row_data = [i, first_image_path, similar_or_duplicate[j]] + list(duplicate_row)
                duplicates_df.loc[len(duplicates_df)] = row_data

    # Move the "Duplicate_Group", "First_Image_Path", and "Duplicate_or_Similar" columns to the beginning
    columns = ['Duplicate_Group', 'First_Image_Path', 'Duplicate_or_Similar'] + list(duplicates_df.columns[3:])
    duplicates_df = duplicates_df[columns]

    return duplicates_df


# Visualize the duplicates using the output of find_near_duplicates()
def visualize_duplicates(duplicates_df, num_groups=5, start_group=1, base_path="../data/raw/Furniture_Data"):
    # Filter the DataFrame based on the start_group and num_groups
    end_group = start_group + num_groups - 1
    duplicates_to_visualize = duplicates_df[
        (duplicates_df["Duplicate_Group"] >= start_group) & (duplicates_df["Duplicate_Group"] <= end_group)]

    # Group the DataFrame by "Duplicate_Group" and get the maximum number of images in any group
    grouped_df = duplicates_to_visualize.groupby("Duplicate_Group")
    num_cols = max(len(group) for _, group in grouped_df)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_groups, num_cols, figsize=(4 * num_cols, 3 * num_groups))

    for i, (group_num, group_df) in enumerate(grouped_df):
        for j, row in enumerate(group_df.itertuples()):
            image_path = f"{base_path}/{row.Path}"  # Combine base_path with the relative path from the DataFrame
            image = Image.open(image_path)
            axes[i, j].imshow(image)

            # Display folder and file name
            folder_name = os.path.basename(os.path.dirname(image_path))
            file_name = os.path.basename(image_path)
            axes[i, j].set_title(f"{folder_name}/{file_name}", fontsize=8)

            axes[i, j].axis('off')

            # Label all images as "Similar" or "Duplicate"
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(file, size=18)
            label = row.Duplicate_or_Similar  # Get the label from the DataFrame # Get the label from the DataFrame
            text_bbox = draw.textbbox((0, 0), label, font=font, anchor="lt")
            text_position = ((image.width - text_bbox[2]) // 2, (image.height - text_bbox[3]) // 2)
            draw.text(text_position, label, font=font, fill=(255, 0, 0), anchor="lt")
            axes[i, j].imshow(image)

        for j in range(len(group_df), num_cols):
            axes[i, j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    plt.show()


# Get the DataFrame containing all duplicates to be deleted
def get_duplicates_to_delete(duplicates_df):
    # Create a list to store the items to delete
    delete_rows = []

    # Get the unique duplicate groups
    duplicate_groups = duplicates_df['Duplicate_Group'].unique()

    # Iterate over each duplicate group
    for group in duplicate_groups:
        # Get the rows for the current duplicate group
        group_df = duplicates_df[duplicates_df['Duplicate_Group'] == group]

        # Check the "Duplicate_or_Similar" column
        if 'Duplicate' in group_df['Duplicate_or_Similar'].values:
            # If "Duplicate" exists, mark all occurrences except the first one for deletion
            duplicates_to_delete = group_df[group_df['Duplicate_or_Similar'] == 'Duplicate'].iloc[1:]
            delete_rows.extend(duplicates_to_delete.to_dict('records'))

    # Create a new DataFrame from the list of rows to delete
    delete_df = pd.DataFrame(delete_rows, columns=duplicates_df.columns)

    return delete_df
