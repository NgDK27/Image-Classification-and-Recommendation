import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def data_count_plot(
        df: pd.DataFrame,
        col: str,
        ax=None,
        horizontal: bool = False,
        title: str = None,
        annotate: bool = True,
        palette=None,
        rotation: float = 0.0
) -> None:
    """
    Create a count plot for a specified column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col (str): The column name for which the count plot is to be created.
        ax (matplotlib.axes.Axes, optional): The axes on which to draw the plot. If not provided, the current axes will be used and displayed.
        horizontal (bool, optional): Whether to create a horizontal count plot. Default is False.
        title (str, optional): The title for the plot. Default is None.
        annotate (bool, optional): Whether to annotate the bars with counts. Default is True.
        palette (str or list of str, optional): The color palette to use for the plot. Default is None.
        rotation (float, optional): The rotation degree for the labels. Does not work with horizontal = True. Default is 0.0

    Returns:
        None
    """

    ax = ax or plt.gca()
    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    sns.countplot(data=df, x=col if not horizontal else None, y=None if not horizontal else col, ax=ax, palette=palette, saturation=1)
    ax.set_xlabel(col) if not horizontal else ax.set_ylabel(col)

    if title:
        ax.set_title(title)

    if annotate:
        ax.bar_label(ax.containers[0], fmt='%.0f')

    if not horizontal:
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=rotation)

    if ax == plt.gca():
        plt.figure(figsize=(10, 10))
        plt.show()

