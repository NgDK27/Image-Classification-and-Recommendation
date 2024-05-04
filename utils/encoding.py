import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def label_encode(df, column_name):
    """
    Label encode a column in the DataFrame.

    Parameters:
    - df: DataFrame
        The DataFrame containing the column to be label encoded.
    - column_name: str
        The name of the column to be label encoded.

    Returns:
    - DataFrame: The DataFrame with the specified column label encoded.
    """
    label_encoder = LabelEncoder()
    df['Encoded_' + column_name] = label_encoder.fit_transform(df[column_name])
    return df


def visualize_normalized_images(df, num_images=5):
    sampled_df = df.sample(n=num_images)
    
    for index, row in sampled_df.iterrows():
        plt.figure(figsize=(3, 3))
        plt.imshow(row['Normalization'])
        plt.title(f'Class: {row["Class"]}')
        plt.axis('off')
        plt.show()