#!/usr/bin/env python
# coding: utf-8

# ### Rahul Sharma

# # Predicition Using Decision Tree Algorithm (Iris dataset)

# In[89]:


import pandas as pd 
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


df= pd.read_csv(r"C:\Users\Rahul\Desktop\iris.csv")
df.head()


# In[91]:


df.count()


# In[92]:


df.info()


# In[93]:


df.dtypes


# In[94]:


df.shape


# In[95]:


df.groupby(['sepal_length','sepal_width']).size()


# In[96]:


df.groupby(['petal_length','petal_width']).size()


# In[97]:


df.groupby('species').size()


# ## Pair Plotting dependent Variables

# In[98]:


import seaborn as sns
sns.pairplot(df,hue='species')


# ## Label Encoding for Dependent variables

# In[99]:


from sklearn import preprocessing as prep
label_encoder = prep.LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])
df['species'].unique()


# In[100]:


df.head()


# ## Declaring Independent and dependent parameters under variables

# In[101]:


x=df[['sepal_length','sepal_width','petal_length','petal_width']]
y=df[['species']]


# ## Creating and Splitting train and test data

# In[102]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)


# # Decision Tree Classification Model

# In[111]:


from sklearn.tree import DecisionTreeClassifier as dt
model=dt(max_depth=10,random_state=100)
model.fit(x_train,y_train)


# # Feature Scaling

# In[112]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test =sc.transform(x_test)
print(x_train)


# # Predicting Model

# In[113]:


predictions=model.predict(x_test)
predictions


# # Classificatin report, Confusion Matrix

# In[114]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[115]:


print(confusion_matrix(y_test,predictions))


# # Predicting the Accuracy of the model

# In[116]:


from sklearn import metrics
print('The accuracy of the DecisionTreeClassifier is:',metrics.accuracy_score(y_test,predictions))


# # Plotting Decision Tree

# In[117]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(50,40))
plot_tree(model,filled=True)


# # Plotting Confusion Matrix

# In[118]:


cfm = (metrics.plot_confusion_matrix(model,x_test,y_test))


# # Thank you for checking ...!!!

# In[ ]:




