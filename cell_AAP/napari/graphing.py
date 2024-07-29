import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional
import matplotlib.pyplot as plt


def read_excel(path: str, sheet_name: str | list[str]) -> pd.DataFrame:
    '''
    Reads an excel file into a pandas dataframe
    ---------------------------------------------
    INPUTS:
        path: str
        sheet_name: str | list[str]
    OUTPUTS:
        df: pd.DataFrame
    '''

    df = pd.read_excel(path, sheet_name=sheet_name)
    return df



def time_in_mitosis(
    df: pd.DataFrame,
    x: str,
    y: str,
    bin: Optional[bool] = False,
    alt_xlabel: Optional[str] = None,
    alt_ylabel: Optional[str] = None,
    title: Optional[str] = None,
):

    """
    Takes in a pandas dataframe along with two variables, x: independent, y: dependent and creates a scatterplot
    -------------------------------------------------------------------------------------------------------------
    INPUTS:
        df: pd.DataFrame
        x: str, must be a coloumn of df
        y: str, must be a coloumn of df
        bin: bool, whether or not to plot binned averages
        alt_xlabel: str
        alt_ylabel: str
        title: str
    OUTPUTS:
        fig: matplotlib.pyplot.plt.subplot object

    """

    sns.set_style("white")
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), sharex=True, sharey=True)
    sns.scatterplot(data=df, x=x, y=y, color="0.8")
    sns.despine()

    (
        ax.set_xlabel(alt_xlabel, fontsize=20, fontweight="bold")
        if alt_xlabel
        else ax.set_xlabel(x, fontsize=15, fontweight="bold")
    )
    (
        ax.set_ylabel(alt_ylabel, fontsize=20, fontweight="bold")
        if alt_ylabel
        else ax.set_ylabel(y, fontsize=15, fontweight="bold")
    )
    if title:
        plt.suptitle(title, fontsize=30, fontweight="bold")

    if bin:
        x_vec = df[x].max()
        bins = np.linspace(df[x].min(), df[x].max(), 11)
        labels = np.linspace(1, 10, 10)
        df["bin"] = pd.cut(df[x], bins=bins, labels=labels)

        binned_dfs = [df[df["bin"] == label] for _, label in enumerate(labels)]
        binned_averages = [
            binned_df[y].mean() for _, binned_df in enumerate(binned_dfs)
        ]
        binned_errors = [
            (binned_df[y].std() / (len(binned_df[y]) + np.finfo(float).eps))
            for _, binned_df in enumerate(binned_dfs)
        ]
        bin_centers = [
            (bins[i] + bins[i + 1]) / 2
            for i, _ in enumerate(bins)
            if i < (len(bins) - 1)
        ]

        plt.errorbar(
            bin_centers,
            binned_averages,
            yerr=binned_errors,
            barsabove=True,
            fmt="o",
            capsize=6,
            c="0",
        )

    return fig
