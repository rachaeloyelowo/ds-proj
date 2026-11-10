# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# reading file
df = pd.read_csv("BEA_Ind-Ind.csv", index_col=0)
print("Data types:", df.dtypes.value_counts())
print("Missing values:", df.isna().sum().sum())

assert df.shape == (71, 71)
assert df.select_dtypes(include="number").shape == (71, 71)


# dataset size
print("Dimensions:", df.shape) # returns dimensions (row by columns) of dataset - 71*71
print("Records:", df.shape[0]) # returns number of records - 71
print("Columns:", df.shape[1]) # returns number of columns - 71
print("Total entries:", df.size) # returns number of total entries in dataset - 5041

print("Datatype size:", df.dtypes) # returns datatype of each column - float64 = 8bytes


# reusable function definitions

def dataset_snapshot(data):
    return data.iloc[:5, :5]


def descriptive_stats(data):
    values = data.to_numpy().flatten()

    return {
        "Mean": np.mean(values),
        "Median": np.median(values),
        "Standard Deviation": np.std(values),
        "Minimum": np.min(values),
        "Maximum": np.max(values)
    }



def frequency_counts(data, bins=50):
    values = data.to_numpy().flatten()

    counts, edges = np.histogram(values, bins=bins)

    for i in range(len(counts)):
        print(f"{edges[i]:.4f} - {edges[i + 1]:.4f}: {counts[i]}")


def plot_histogram(data, filename="histogram.png", max_value=None):
    values = data.to_numpy().flatten()

    plt.figure(figsize=(9, 5))

    if max_value is not None:
        values = values[values <= max_value]

        sns.histplot(
            values,
            bins=np.arange(0, max_value + 0.002, 0.002)
        )

        plt.xticks(np.arange(0, max_value + 0.01, 0.01))

    else:
        sns.histplot(values, bins=50)

    if max_value is not None:
        plt.title(f"Distribution of Requirement Coefficients (0–{max_value})")
    else:
        plt.title("Distribution of All Requirement Coefficients")
    plt.xlabel("Requirement Coefficient")
    plt.ylabel("Frequency")

    mean = np.mean(values)

    sns.histplot(values, bins=50)

    plt.axvline(
        mean,
        color="red",
        linestyle="--",
        label=f"Mean: {mean:.4f}"
    )
    median = np.median(values)

    plt.axvline(
        median,
        color="green",
        linestyle="--",
        label=f"Median: {median:.4f}"
    )

    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_pairplot(data, filename="pairplot.png"):
    selected = data.iloc[:, :4].copy()

    plot = sns.pairplot(selected, diag_kind="hist")

    plot.fig.suptitle(
        "Pairwise Comparison of Industry Requirements",
        y=1.02
    )

    plot.savefig(filename, dpi=300)
    plt.show()


def plot_zscores(data, filename="zscores.png"):
    values = data.to_numpy().flatten()

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        print("Cannot calculate z-scores: standard deviation is zero.")
        return

    z_scores = (values - mean) / std

    plt.figure(figsize=(9, 5))
    sns.histplot(z_scores, bins=50)

    plt.title("Z-Score Distribution of Requirement Coefficients")
    plt.xlabel("Z-Score")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def cross_industry_values(data):
    values = data.to_numpy()

    if values.shape[0] != values.shape[1]:
        raise ValueError("Expected a square industry matrix.")

    mask = ~np.eye(values.shape[0], dtype=bool)

    return values[mask]

def plot_cross_industry_histogram(data, filename="cross_industry.png"):
    off_diagonal = cross_industry_values(data)

    plt.figure(figsize=(9, 5))

    off_diagonal = off_diagonal[off_diagonal <= 0.1]

    sns.histplot(
        off_diagonal,
        bins=np.arange(0, 0.102, 0.002)
    )
    plt.xlim(0, 0.1)
    plt.xticks(np.arange(0, 0.101, 0.01))

    plt.title("Distribution of Cross-Industry Requirements (0–0.1)")
    plt.xlabel("Requirement Coefficient")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# function calls
print("Dataset Snapshot:")
print(dataset_snapshot(df))

print("\nDescriptive Statistics:")
print(descriptive_stats(df))


print("\nFrequency Counts:")
frequency_counts(df)

plot_histogram(df, "bea_full_histogram.png")

plot_histogram(
    df,
    "bea_zoomed_histogram.png",
    max_value=0.1
)

plot_pairplot(df, "bea_pairplot.png")

plot_zscores(df, "bea_zscores.png")

off_diagonal = cross_industry_values(df)

print("\nCross-industry coefficients:", len(off_diagonal))
print("Cross-industry mean:", np.mean(off_diagonal))
print("Cross-industry median:", np.median(off_diagonal))

plot_cross_industry_histogram(
    df,
    "bea_cross_industry_histogram.png"
)