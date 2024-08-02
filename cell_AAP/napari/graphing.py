import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import scipy.optimize as opt
from typing import Optional
import matplotlib.pyplot as plt



def logistic(x, a, b, c, d):
    "General logistic function"
    return a / (1. + np.exp(-c * (x - d))) + b

def linear(x, a, m, b):
    "General linear function"
    return m*(x- a) + b

def fit_wrapper(func_name: str, x: npt.NDArray, y: npt.NDArray):
    "Takes a function name and returns the optimized function corresponding function"

    match func_name.split():
        case ["logistic"]:
            params, _= opt.curve_fit(logistic, x, y, method = 'trf')
            print(params)
            return logistic(x, *params)
        case ["linear"]:
            params, _=  opt.curve_fit(linear, x, y)
            return linear(x, *params)
        case _:
            raise ValueError(f"Value of func_name = {func_name} was not a valid option")

def time_in_mitosis(
    df: pd.DataFrame,
    x: str,
    y: str,
    bin: Optional[bool] = False,
    alt_xlabel: Optional[str] = None,
    alt_ylabel: Optional[str] = None,
    title: Optional[str] = None,
    fit: str = "n",
    xlim:  Optional[list] = None,
    ylim: Optional[list] = None,
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
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    sns.scatterplot(data=df, x=x, y=y, color="0.8", ax=ax)

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

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    df = df[~(df[y] <= 10)].copy()

    if bin:
        labels = np.linspace(1, 20, 20)
        df["bin"], bins = pd.qcut(df[x], q = 20, labels=labels, retbins = True)
        binned_dfs = [df[df["bin"] == label] for _, label in enumerate(labels)]
        binned_averages = [
            binned_df[y].mean() for _, binned_df in enumerate(binned_dfs)
        ]
        binned_errors = [
            (binned_df[y].std() / (len(binned_df[y]) + np.finfo(float).eps))
            for _, binned_df in enumerate(binned_dfs)
        ]

        bins = np.asarray(bins)
        binned_averages = np.asarray(binned_averages)
        bin_centers = [
            (bins[i] + bins[i + 1]) / 2
            for i, _ in enumerate(bins)
            if i < (len(bins) - 1)
        ]

        ax.errorbar(
            bin_centers,
            binned_averages,
            yerr=binned_errors,
            barsabove=True,
            fmt="o",
            capsize=6,
            c="0",
        )

        if fit != "n":
            try:
                func = fit_wrapper(fit, np.asarray(bin_centers), np.asarray(binned_averages))
                curve_data = pd.DataFrame({'curve_x' : np.asarray(bin_centers), 'curve_y' : func},)
                sns.lineplot(data = curve_data, x = 'curve_x', y = 'curve_y', ax = ax, color = 'b')
            except Exception as error:
                print(error)

    return fig
