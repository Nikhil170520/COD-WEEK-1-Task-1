#!/usr/bin/env python
# coding: utf-8

# In[20]:


#TASK ONE :- EXPLORATORY DATA ANALYSIS(EDA) 
import pandas as pd
import numpy as np


# In[13]:


#Load datset
df=pd.read_csv(r'D:\CardioGoodFitness.csv')


# In[14]:


df


# In[15]:


data=np.genfromtxt("D:\CardioGoodFitness.csv",delimiter=",",skip_header=1)


# In[16]:


data


# In[21]:


# data Characteristics
df.head()


# In[22]:


df.tail()


# In[23]:


df.describe()


# In[24]:


df.isnull().sum()


# In[28]:


df.info()


# In[25]:


#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


plt.scatter('Miles','Education')
plt.xlabel('Miles')
plt.ylabel('Education')
plt.title('Simple Scatter Plot')
plt.show()


# In[56]:


#Data Distribution
df.hist(bins=100)
plt.title('Simple Distribution')
plt.show()


# In[34]:


# Find Outliers
#using Box Plot
sns.boxplot(data=df,x='Miles',showmeans=True)
plt.title('Box Plot oF Miles')
plt.show()


# In[36]:


#finding Outliers
sns.scatterplot(x='Miles',y='Usage',data=df)
plt.xlabel('Miles')
plt.ylabel('Usage')
plt.title('Relation between Miles and Usage')
plt.show()


# In[64]:


# Data Visualize
# Histogram , Scatter , Heatmap

# Histogram
plt.hist(x=(df['Age'],df['Usage'],df['Miles']),bins=10)
plt.title('Histogram Chart')
plt.show()


# In[68]:


# Scatter PLot
plt.scatter(x=df['Age'],y=df['Miles'])
plt.xlabel('Age')
plt.ylabel('Miles')
plt.title('Scatter Plot')
plt.show()


# In[50]:


# HeatMap 
from sklearn.metrics import confusion_matrix as cn


# In[70]:


con=cn(df['Miles'],df['Age'])


# In[73]:


sns.heatmap(con,annot=True,cmap='tab20c')
plt.title('Heatmap')
plt.show()


# In[ ]:




