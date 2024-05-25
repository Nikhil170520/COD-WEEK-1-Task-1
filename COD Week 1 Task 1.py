#TASK ONE :- EXPLORATORY DATA ANALYSIS(EDA) 
import pandas as pd
import numpy as np
#Load datset
df=pd.read_csv(r'D:\CardioGoodFitness.csv')
df
data=np.genfromtxt("D:\CardioGoodFitness.csv",delimiter=",",skip_header=1)
data
# Data Characteristics
df.head()
df.tail()
df.describe()
df.isnull().sum()
df.info()
#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter('Miles','Education')
plt.xlabel('Miles')
plt.ylabel('Education')
plt.title('Simple Scatter Plot')
plt.show()
#Data Distribution
df.hist(bins=100)
plt.title('Simple Distribution')
plt.show()
# Find Outliers
#using Box Plot
sns.boxplot(data=df,x='Miles',showmeans=True)
plt.title('Box Plot oF Miles')
plt.show()
#finding Outliers
sns.scatterplot(x='Miles',y='Usage',data=df)
plt.xlabel('Miles')
plt.ylabel('Usage')
plt.title('Relation between Miles and Usage')
plt.show()
# Data Visualize
# Histogram , Scatter , Heatmap
# Histogram
plt.hist(x=(df['Age'],df['Usage'],df['Miles']),bins=10)
plt.title('Histogram Chart')
plt.show()
# Scatter PLot
plt.scatter(x=df['Age'],y=df['Miles'])
plt.xlabel('Age')
plt.ylabel('Miles')
plt.title('Scatter Plot')
plt.show()
# HeatMap 
from sklearn.metrics import confusion_matrix as cn
con=cn(df['Miles'],df['Age'])
sns.heatmap(con,annot=True,cmap='tab20c')
plt.title('Heatmap')
plt.show()




