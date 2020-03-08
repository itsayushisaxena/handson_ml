#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data.head()


# In[2]:


# This reveals we have 891 instances of which some instances contain null values.
train_data.info()


# In[3]:


train_data.describe()

# The mean age was less than 29 years and only 38% people survied.


# In[4]:


# Let's check what our target variable is:

train_data["Survived"].value_counts()

# So it is indeed 0 or 1.


# In[5]:


# Now let's take a quick look at all categorical attributes:
train_data["Pclass"].value_counts()


# In[6]:


train_data["Sex"].value_counts()


# In[7]:


train_data["Embarked"].value_counts()


# In[8]:


# Now let's build our preprocessing pipelines. We will reuse the DataFrameSelector we built in the second chapter 
# to select specific attributes from the DataFrame:

from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns since Scikit-Learn doesn't handle DataFrames yet.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[9]:


# Let's build the pipeline for numerical attributes:

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy="median"))
])


# In[10]:


num_pipeline.fit_transform(train_data)


# In[11]:


#We will also need an imputer for the string categorical columns (the regular SimpleImputer does not work 
# on those):
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[12]:


from sklearn.preprocessing import OneHotEncoder


# In[13]:


# Now we can build the pipeline for the categorical attributes:

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])


# In[14]:


cat_pipeline.fit_transform(train_data)


# In[15]:


# Now let's join the numerical and categorical pipelines:

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[16]:


X_train = preprocess_pipeline.fit_transform(train_data)
X_train


# In[17]:


y_train = train_data["Survived"]


# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()

