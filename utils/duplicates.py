# Imports
import imagehash
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from annoy import AnnoyIndex
from matplotlib import font_manager
from tqdm import tqdm
# noinspection PyUnresolvedReferences
import swifter

# Fonts for visualization
font = font_manager.FontProperties(family='sans-serif', weight='bold')
file = font_manager.findfont(font)

# Setting default variables
BASE_PATH = "../data/raw/Furniture_Data"
HASH_SIZE = 16  # Larger hash size means better differentiation between visually similar images
DIFFERENCE_THRESHOLD = 1  # Difference in bits between hashes of images.
# Larger DIFFERENCE_THRESHOLD means more visually similar images are counted as duplicates.
CLOSEST_DUPLICATES_THRESHOLD = 10  # How many closest items to find during get_nns_by_item().
# Smaller CLOSEST_DUPLICATES_THRESHOLD means less duplicates to find, lessening the run time.


# Calculate perceptual hash
def calculate_phash_row(row, base_path=BASE_PATH):
    image_path = f"{base_path}/{row['Path']}"
    try:
        image = Image.open(image_path)
        phash_value = str(imagehash.phash(image, hash_size=16))
        return phash_value
    except Exception as e:
        print(f"Error processing image: {image_path}")
        print(f"Error message: {str(e)}")
        return None


# Calculate perceptual hash
def calculate_phash(df):
    print("Calculating perceptual hash for all images...")
    df['Hash'] = df.swifter.apply(calculate_phash_row, axis=1)
    return df


# Convert hash string to binary
def hex_to_binary(hex_string, hash_size=HASH_SIZE):
    num_bits = hash_size * hash_size
    return bin(int(hex_string, 16))[2:].zfill(num_bits)


# Build approximate nearest neighbors tree, then identifying and grouping duplicates
def find_duplicate_groups(df, difference_threshold=DIFFERENCE_THRESHOLD, hash_size=HASH_SIZE,
                          closest_duplicates_threshold=CLOSEST_DUPLICATES_THRESHOLD):
    print("Building approximate nearest neighbors using Annoy...")

    # Create an Annoy index
    index = AnnoyIndex(hash_size * hash_size, metric='hamming')

    # Add binary hashes to the index
    for i, hex_hash in enumerate(df['Hash']):
        binary_hash = hex_to_binary(hex_hash, hash_size)
        index.add_item(i, [int(bit) for bit in binary_hash])

    # Build the index
    index.build(10, n_jobs=-1)

    # Find duplicate and near-duplicate groups
    duplicate_groups = []
    visited = set()

    print("Identifying and grouping duplicates...")
    for i in tqdm(range(len(df))):
        if i in visited:
            continue

        group = [i]
        visited.add(i)

        # Find near-duplicates within the distance threshold
        neighbors, distances = index.get_nns_by_item(i, closest_duplicates_threshold, include_distances=True)
        for j, distance in zip(neighbors, distances):
            if j != i and j not in visited:
                if distance <= difference_threshold:
                    group.append(j)
                    visited.add(j)

        duplicate_groups.append(group)

    return duplicate_groups


# Labels group number, uses output of find_duplicate_groups()
def assign_group_labels(df, duplicate_groups):
    print("Labelling duplicates group (non-duplicates are not included in result)...")

    group_labels = [0] * len(df)

    group_id = 1
    for group in tqdm(duplicate_groups):
        if len(group) > 1:
            for duplicate_index in group:
                group_labels[duplicate_index] = group_id
            group_id += 1

    df['Group'] = group_labels

    # Remove non-duplicated rows
    df = df[df['Group'] != 0]

    # Move the "Group" column to the start
    column_order = ['Group'] + [col for col in df.columns if col != 'Group']
    df = df[column_order]

    # Group the rows by the "Group" values
    df = df.sort_values('Group')

    return df


# Labels duplicate type, uses output of assign_group_labels()
def assign_duplicate_types(df):
    print("Labelling duplicates type...")

    df['Duplicate_Type'] = ''

    for group_id in tqdm(df['Group'].unique()):
        group_rows = df[df['Group'] == group_id]

        if group_rows['Class'].nunique() == 1:
            style_counts = group_rows['Style'].value_counts()
            duplicate_styles = style_counts[style_counts > 1].index

            for style in duplicate_styles:
                duplicate_mask = (df['Group'] == group_id) & (df['Style'] == style)
                df.loc[duplicate_mask, 'Duplicate_Type'] = 'Duplicate'

            similar_mask = (df['Group'] == group_id) & (df['Duplicate_Type'] != 'Duplicate')
            df.loc[similar_mask, 'Duplicate_Type'] = 'Similar'
        else:
            df.loc[df['Group'] == group_id, 'Duplicate_Type'] = 'Inspect'

    # Move the "Duplicate_Type" column to be the second column
    column_order = ['Group', 'Duplicate_Type'] + [col for col in df.columns if col not in ['Group', 'Duplicate_Type']]
    df = df[column_order]

    return df


# Complete duplicate finding pipeline
def prepare_duplicates(df):
    duplicates = calculate_phash(df)
    duplicates_groups = find_duplicate_groups(df)
    duplicates = assign_group_labels(duplicates, duplicates_groups)
    duplicates = assign_duplicate_types(duplicates)
    return duplicates


# Visualize
def visualize_duplicates(duplicates_df, num_groups=5, start_group=None, title=None, base_path=BASE_PATH):
    # Get the unique group numbers
    unique_groups = sorted(duplicates_df["Group"].unique())

    # If start_group is not specified, use the first group number
    if start_group is None:
        start_group = unique_groups[0]

    # Find the index of the start_group in the unique_groups list
    start_index = unique_groups.index(start_group)

    # Get the group numbers to visualize based on start_group and num_groups
    groups_to_visualize = unique_groups[start_index:start_index + num_groups]

    # Filter the DataFrame based on the groups to visualize
    duplicates_to_visualize = duplicates_df[duplicates_df["Group"].isin(groups_to_visualize)]

    # Group the DataFrame by "Group" and get the maximum number of images in any group
    grouped_df = duplicates_to_visualize.groupby("Group")
    num_cols = max(len(group) for _, group in grouped_df)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_groups, num_cols, figsize=(4 * num_cols, 3 * num_groups))

    fig.suptitle(title, y=1.0)

    for i, (group_num, group_df) in enumerate(grouped_df):
        for j, row in enumerate(group_df.itertuples()):
            image_path = f"{base_path}/{row.Path}"  # Combine base_path with the relative path from the DataFrame
            image = Image.open(image_path)
            axes[i, j].imshow(image)

            # Display folder and file name
            axes[i, j].set_title(f"{row.Path[:25]}...{row.Path[-10:]}", fontsize=8)

            axes[i, j].axis('off')

            # Label all images as "Similar" or "Duplicate"
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(file, size=18)
            label = row.Duplicate_Type  # Get the label from the DataFrame
            text_bbox = draw.textbbox((0, 0), label, font=font, anchor="lt")
            text_position = ((image.width - text_bbox[2]) // 2, (image.height - text_bbox[3]) // 2)
            draw.text(text_position, label, font=font, fill=(255, 0, 0), anchor="lt")
            axes[i, j].imshow(image)

        for j in range(len(group_df), num_cols):
            axes[i, j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    plt.show()


# Get duplicates rows to delete (one of each "Style" of rows marked as "Duplicate" within a group will be kept)
def get_duplicates_to_delete(df):
    print("Identifying rows to delete...")

    rows_to_delete = []

    for group_id in tqdm(df['Group'].unique()):
        group_rows = df[df['Group'] == group_id]
        duplicate_rows = group_rows[group_rows['Duplicate_Type'] == 'Duplicate']

        if len(duplicate_rows) > 0:
            style_counts = duplicate_rows['Style'].value_counts()
            rows_to_keep = duplicate_rows.drop_duplicates(subset=['Style'])

            for _, row in duplicate_rows.iterrows():
                if row['Style'] in style_counts[style_counts > 1].index and row.name not in rows_to_keep.index:
                    rows_to_delete.append(row.name)

    return df.loc[rows_to_delete]


# Remove rows
def remove_rows(original_df, rows_to_delete):
    print("Removing rows...")

    # Get the indices of the rows to delete
    indices_to_delete = rows_to_delete.index

    # Create a new DataFrame by removing the rows to delete from the original DataFrame
    cleaned_df = original_df.drop(indices_to_delete)

    # Reset the index of the cleaned DataFrame
    cleaned_df.reset_index(drop=True, inplace=True)

    return cleaned_df
