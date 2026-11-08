# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# reading file
df = pd.read_csv("BEA_Ind-Ind.csv", index_col=0)


# dataset size
print("Dimensions:", df.shape) # returns dimensions (row by columns) of dataset - 71*71
print("Records:", df.shape[0]) # returns number of records - 71
print("Columns:", df.shape[1]) # returns number of columns - 71
print("Total entries:", df.size) # returns number of total entries in dataset - 5041

print("Datatype size:", df.dtypes) # returns datatype of each column - float64 = 8bytes


# dataset snapshot

# dataset sample statistics and plots


values = df.to_numpy().flatten() # convert 2D df to 2D np arr then to a 1D arr

# using bins because each entry can technically be any val

# cleaning to only use values less than 0.1 since that is where most of the data is located
# outliers near 1; represents the intersection of an industry with itself
cleaned_val = values[values <= 0.1]

plt.figure(figsize=(9, 5))

sns.histplot(
    cleaned_val,
    bins=np.arange(0, 0.101, 0.002)
)

plt.xticks(np.arange(0, 0.101, 0.01))

plt.title("Distribution of BEA Requirement Coefficients (0–0.1)")
plt.xlabel("Requirement Coefficient")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("bea_histogram.png", dpi=300)
plt.show()