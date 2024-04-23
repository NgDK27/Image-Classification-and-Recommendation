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
        if horizontal:
            for p in ax.patches:
                ax.annotate(f'{p.get_width():.0f}', (p.get_width() / 2., p.get_y() + p.get_height() / 2.), ha='center',
                            va='center', xytext=(0, 0), textcoords='offset points')
        else:
            ax.bar_label(ax.containers[0], fmt='%.0f')

    if ax is None:
        plt.show()