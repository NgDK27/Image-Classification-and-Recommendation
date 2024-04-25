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
) -> None:
    """
    Create a count plot for a specified column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col (str): The column name for which the count plot is to be created.
        ax (matplotlib.axes.Axes, optional): The axes on which to draw the plot. If not provided, the current axes will be used.
        horizontal (bool, optional): Whether to create a horizontal count plot. Default is False.
        title (str, optional): The title for the plot. Default is None.
        annotate (bool, optional): Whether to annotate the bars with counts. Default is True.
        palette (str or list of str, optional): The color palette to use for the plot. Default is None.

    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()

    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    if horizontal:
        sns.countplot(y=df[col], ax=ax, palette=palette, saturation=1)
        ax.set_ylabel(col)
    else:
        sns.countplot(x=df[col], ax=ax, palette=palette, saturation=1)
        ax.set_xlabel(col)

    if title is not None:
        ax.set_title(title)

    if annotate:
        # if horizontal:
        #     for p in ax.patches:
        #         ax.annotate(f'{p.get_width():.0f}', (p.get_width() / 2., p.get_y() + p.get_height() / 2.), ha='center',
        #                     va='center', xytext=(0, 0), textcoords='offset points')
        # else:
        ax.bar_label(ax.containers[0], fmt='%.0f')

    if ax is None:
        plt.show()
