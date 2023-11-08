# Problem Statement 1

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

# importing the dataset

Country = pd.read_csv('Assignment 5 Datasets/Country-data.csv')
Country.info()

# Reading the dataset

Country_num = Country.drop(['country'], axis=1)
Country_num.head()

# Scale the data using the Standard Scaler to create a scaled DataFrame

standard_scaler = StandardScaler()
scaled = standard_scaler.fit_transform(Country_num)

scaled_df = pd.DataFrame(scaled, columns=Country_num.columns)

scaled_df.head()



# Plotting dendograms with the complete linkage method

# Compute the linkage
Z = linkage(scaled_df, method='complete')  # use 'complete' linkage

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Complete Linkage Dendrograms")
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()


# Creating cluster labels using cut tree

labels = cut_tree(Z, n_clusters=3)

scaled_df_1 =scaled_df.copy()

# Creating a new dataframe to store the cluster labels

scaled_df_1['Hierarchical_cluster_labels'] = labels

scaled_df_1

# Performing the 4 compenents PCA on the dataframe

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(scaled_df)

# Convert the principal components into a dataframe
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])

principalDf ['Hierarchical_cluster_labels'] = labels

principalDf

# Now, from final the DataFrame, analyze how low GDP rate corresponds to the chil mortality rate around the world

# 3. Visual Analysis
plt.figure(figsize=(12, 7))
sns.scatterplot(data = scaled_df_1, x = 'gdpp', y ='child_mort', hue='Hierarchical_cluster_labels')
plt.title("Child Mortality vs GDP for Low GDP Countries")
plt.xlabel("GDP")
plt.ylabel("Child Mortality Rate")
plt.grid(True)
plt.show()


# Problem 2

from sklearn.cluster import KMeans

# Load the data into a DataFrame
df= pd.read_csv('Assignment 5 Datasets/Credit Card Customer Data.csv')

df.info()

# Scale the data using the Standard Scaler
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df.head()