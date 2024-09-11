#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


df=pd.read_csv('C:/Users/Lenovo/OneDrive/Bureau/Medical Visualization/Medical-Data-Visualizer/medical_examination.csv')
df


# In[2]:


df.info()


# In[3]:


df.isnull().sum()


# In[4]:


df.duplicated().sum()


# -------------------Visualizer----------------------------------------------

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# In[7]:


df_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'alco', 'active', 'smoke'])


# In[8]:


g = sns.catplot(x='variable', hue='value', col='cardio', data=df_long, kind='count', height=5, aspect=1.2)
g.set_axis_labels("Variable", "Count")
g.set_titles("Cardio = {col_name}")
g.despine(left=True)
plt.show()


# In[9]:


df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)


# In[10]:


df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# In[11]:


df_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'alco', 'active', 'smoke', 'overweight'])


# In[12]:


g = sns.catplot(x='variable', hue='value', col='cardio', data=df_long, kind='count', height=5, aspect=1.2)
g.set_axis_labels("Variable", "Count")
g.set_titles("Cardio = {col_name}")
g.despine(left=True)
plt.show()


# In[13]:


df = df[(df['ap_lo'] <= df['ap_hi'])]
df = df[(df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975))]
df = df[(df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]


# In[14]:


df_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'alco', 'active', 'smoke', 'overweight'])


# In[15]:


df


# In[16]:


import numpy as np

corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix,mask=mask,  annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=.5, center=0)
plt.title('Correlation Matrix')
plt.show()

