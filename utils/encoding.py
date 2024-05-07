import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def one_hot_encode(df, column_name):
    """
    Perform one-hot encoding on a specified column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame on which to perform encoding.
    column_name (str): The name of the column to encode.
    
    Returns:
    pd.DataFrame: A DataFrame with the original column replaced by one-hot encoded columns.
    """
    return pd.get_dummies(df, columns=[column_name])

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