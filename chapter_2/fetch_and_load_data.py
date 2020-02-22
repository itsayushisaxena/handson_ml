#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Here is the function to fetch the data from the github repository. It is useful in particular if 
# data changes regularly, as it allows you to write a small script that you can run whenever you need 
# to fetch the latest data
import os
import tarfile
from six.moves import urllib


# In[2]:


# DOWNLOAD_ROOT is the which contains the .tgz file of data (write it as it is)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/" 
# HOUSING_PATH is the path where .tgz file will be downloaded in your system (it can be changed)
HOUSING_PATH = "/home/cipher/aston/datum/O'reilly_handson_ml/chapter_2"
# HOUSING_URL is the download path of the .tgz file (write it as it is)
HOUSING_URL = DOWNLOAD_ROOT + "housing.tgz"


# In[3]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[4]:


fetch_housing_data(HOUSING_URL, HOUSING_PATH)


# In[5]:


# Load the data using pandas
import pandas as pd


# In[6]:


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[7]:


housing = load_housing_data()
housing.head()


# In[8]:


# info() method is useful to get a quick description of data
housing.info()

# Here we can clearly see there are 20640 instances in the dataset, of which total_bedrooms attribute has
# only 20,433 instances meaning 207 instances are missing this feature. Also ocean_proximity attribute type is 
# object & when you look at the top five rows, you probably noticed that the values in ocean_proximity column 
# were repetitive which means that it is a categorical attribute.


# In[9]:


# Now to find out what categories exist you can use
housing["ocean_proximity"].value_counts()


# In[10]:


# Another method describe() shows a summary of the numerical attributes
housing.describe()


# In[11]:


# Another way to get a feel of the type of data you are dealing with is to plot a histogram for each 
# numerical attribute
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[12]:


# The given dataset should be split into train_set and test_set which are used to train the model and test 
# the model respectively. Scikit-Learn provides train_test_split function to do so.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Here test_size ensures 20% of the dataset is picked up randomly as test_set
# The test_set should be representative of the whole dataset, which is ensured by Scikit-Learn. This is called 
# stratified sampling.


# In[13]:


# median_income is a very important attribute to predict median housing prices. We need to ensure that the 
# test_set is representative of the various categories of income in whole dataset. Since median_income is a 
# continuos numerical attribute, you first need to create and income attribute.
# Looking at the median_income histogram we see most median_income values are clustered around $20,000-$50,000,
# but some median_income go far beyond $60,000.
import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# This creates an income category attribute by dividing the median_income by 1.5 (to limit the number of 
# income categories), and rounding up using ceil (to have discrete categories), and then merging all the 
# categories greater than 5 into category 5.

housing.head()


# In[14]:


# Now to do stratified sampling based on income category is to use Scikit_Learn's StratifiedShuffleSplit class.
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# For more information, see Readme.md 
# To see if it worked as expected, you can start by lookig at the income category proportions in full dataset
housing["income_cat"].value_counts()/len(housing)


# In[15]:


# Now to remove the income_cat attribute so the data is back to its original state:
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[16]:


# Our train data looks like
strat_train_set.head()


# In[17]:


# Let's get into more detail of the dataset but first make a copy of the train dataset.
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")

# This looks like California but it is hard to see any pattern.


# In[18]:


# Setting the alpha option makes it easier to visualize the places where there is high density.
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# The alpha blending value, between 0 (transparent) and 1 (opaque).
# You can clearly see the high-density areas, namely the Bay Area and around Los Angeles and San Diego,
# plus a long line of fairly high density in the Central Valley.


# In[19]:


# The radius of each circle represents the district's population (option s), and the color represents the price
# (option c). Here we will use a predefined color map (option cmap) called jet, which ranges from 
# blue (low values) to red (high values).
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, 
              label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"),
              colorbar=True
             )
plt.legend()

# This image tell you that the housing prices are very much related to the location ad the population density.


# In[20]:


# To find out the standard correation coefficient (also called Pearson's r) use
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Here we see a strong positive correlation between median_house_value and median_income i.e. median house 
# value tends to go up when the median income goes up. 
# There is a small negative correlation between the latitude and the median house value i.e. prices have a slight
# tendency to go down as you go north.


# In[21]:


# Experimenting with Attribute Combinations
# Total number of bedrooms in a district is not very useful if you don't know how many households there are. What
# you really want is the number of rooms per household.
# Similarly, the total number of bedrooms and the population per household also seems like an interesting 
# attribute combination to look at
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# And now let's look at correlation matrix again:
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# The new bedrooms_per_room attribute is much more correlated with the median house value than the total number
# of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio tend to be more expensive.


# In[22]:


# Now let's separate the predictors and the labels
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[23]:


# Scikit-Learn provides a handy class to take care of missing values: SimpleImputer.
# First, you need to create an Imputer instance, specifying that you want to replace each attribute's missing 
# values with the median of that attribute.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# Since meadian can only be computed on numerical atttributes, we need to create a copy of the data without the 
# text attribute ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)

# Now you can fit the imputer instance to the training data using the fit() method:
imputer.fit(housing_num)

# The imputer has simply computed the median of each attribute and stored the result in its statistics_ 
# instance variable.
imputer.statistics_


# In[24]:


# Now you can use this "trained" imputer to transform the training set by replacing missing values by the  
# learned medians. The result is a plain Numpy array.
X = imputer.transform(housing_num)


# In[25]:


# Earlier we left out the categorical attribute ocean_proximity because it is a text attribute so we cannot 
# compute its median. Let's convert these text labels to numbers using Scikit-Learn's transformer for this 
# task called LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# In[26]:


# Although Scikit-Learn provied many useful transformers, you will need to write yourown. All you need is to 
# create a class and implement three methods: fit() (returning self), transform(), and fit_transform().
# You can get the last one for free by simply adding TransformerMixin as base class. Also, if you add 
# BaseEstimator as a base class (and avoid *args and **kargs in your constructor) you will get two extra methods 
# (get_params() and set_params()) that will be useful for automatic hyperparameter tuning.
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):  # no *args pr **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[27]:


# There are many data transformations steps that need to be executed in the right order. Fortunately, Scikit-Learn
# provides the Pipeline class to help with such sequences of transformations. Here is a small pipeline for the 
# numerical attributes:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
 ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# Here StandardScaler is used for standarization for the purpose of feature scaling i.e. to get all attributes
# to have the same scale. First it subtracts the mean value (so standarized values always have a zero mean), 
# and then it divides by the variance so that the resulting distribution have zero variance.


# In[28]:


# Now it would be nice if we could feed a Pandas DataFrame directly into our pipeline, instead of having to first
# manually extract the numerical columns into a NumPy array.
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Our DataFrameSelector will transform the data by selecting the desired attributes, dropping the rest, and 
# converting the resulting DataFrame to a NumPy array.


# In[29]:


# By default LabelBinarizer takes two input now after update to Scikit-Learn 0.18.0 because they said 
# LabelBinarizer is meant fot labels only and not for features. To use LabelBinarizer here, you can create you own
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)


# In[30]:


# You can write another pipeline for the categorical attributes as well by simply selecting the categorical 
# attributes using a DataFrameSelector and then applying a LabelBinarizer.
from sklearn.preprocessing import LabelBinarizer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
 ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
])


# In[31]:


# To join above two pipelines use Scikit-Learn's FeatureUnion class
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# And you can run the whole pipeline simply by:
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape


# In[32]:


# Let's train a Linear Regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# 

# In[33]:


# Done! We now have a working Linear Regression model. Let's try it out on a few instances from training set.
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))

# It works, although the predictions are not exactly accurate.


# In[34]:


print("Labels:", list(some_labels))


# In[35]:


# Let's measure this regression model's RMSE on the whole training set using Scikit-Learn's mean_squared_error 
# function
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# This is not clearly a good score as median_housing_values range between $120,000 and $265,000, so a 
# prediction error of $68,628 is not very satisfying. This is an example of a model underfitting the training 
# data. To overcome this we can use a more powerful model, feed the training algorithm with better features, or
# to reduce the constraints on the model.


# In[36]:


# Let's try a more complex model. Let's train a DecisionTreeRegressor.
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[37]:


# Let's evaluate it on the training set
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# Wait, what? No error at all? Model has badly overfit the data.
# We need to use part of the training set for training, and part for model validation.


# In[38]:


# One way to evaluate Decision Tree model would be to use the train_test_split function to split the training set
# into a smaller training set and a validation set, then train your mdoels against the smaller training set and
# evaluate them against the validation set.
# A great alternative is to use Scikit-Learn's cross-validation feature. 
# Following code performs K-fold cross-validation: it randomly splits the training set into 10 distinct subsets
# called folds, then it trains and evaluates the Decision Tree model 10 times picking a different fold for 
# evaluation every time and training on the other 9 folds.
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

# For moer info see Readme.md
# The result is an array containing the 10 evaluation scores. Let's look at the results:
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    
display_scores(tree_rmse_scores)

# For more information see Readme.md
# Now the Decision Tree doiesn't look as good as it did earlier. In fact, it seems to perform worse than the 
# Linear Regression model. The Decision Tree has a score of approximately 71,379, generally ± 2,458.


# In[39]:


# Let's compute the same scores for the Linnear Regression model
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[40]:


# Let's try one last mdoel now: the RandomForestRegressor. Random Forest works by training many Decision Trees 
# on random subsets of the features, then averaging out their predictions. Building a model on top of many other
# models is called Ensemble Learning.
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

# Let's compute the scores for RandomForestRegressor model
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)

# This is much better. However, note that the score on the training set is still much lower than on the validation
# sets, meaning that the model is still overfitting the training set.


# In[46]:


# Let's try a Support Machine Vector regressor with linear kernel
from sklearn.svm import SVR
svr_regl = SVR(kernel="linear")
svr_regl.fit(housing_prepared, housing_labels)

scores = cross_val_score(svr_regl, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
svrl_rmse_scores = np.sqrt(-scores)
display_scores(svrl_rmse_scores)

# This is the best till now. See 


# In[42]:


# Let's try another Support Machine Vector regressor with rbf kernel
from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(housing_prepared, housing_labels)

scores = cross_val_score(svr_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
svr_rmse_scores = np.sqrt(-scores)
display_scores(svr_rmse_scores)


# In[76]:


# You should save every model you experiment with, so you can come back easily to any model you want.
from sklearn.externals import joblib

joblib.dump(lin_reg, "lin_reg.pkl")
joblib.dump(tree_reg, "tree_reg.pkl")
joblib.dump(forest_reg, "forest_reg.pkl")
joblib.dump(svr_reg, "svr_reg.pkl")      # svr_reg for SVM with default kernel i.e., kernel="rbf"
joblib.dump(svr_regl, "svr_regl.pkl")    # svr_regl for SVM with kernel="linear"


# In[74]:


# Now to load any model type
my_model_loaded = joblib.load("forest_reg.pkl")
my_model_loaded 


# In[80]:


# To fine tune your model i.e., to get a great combination of hyperparameter values, you can use Scikit-Learn's
# GridSearchCV. All you need to do is tell it which hyperparameters you want it to experiment with, and what 
# values to try out, and it will evaluate all the possible combinations of hyperparameter values, using
# cross-validation.
# The following code searches for the best combination of hyperparameter values for the RandomForestRegressor:
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# This param_grid tells Scikit-Learn to first evaluate all 3 X 4 = 12 combinations of n_estimators and 
# max_features hyperparameter values specified in the first dict, then try all 2 X 3 = 6 combinations of
# hyperparameter values in the second dict, but this time with the bootstrap hyperparameter set to False instead
# of True (which is the default value of this parameter).
# All in all, the grid will explore 12 + 6 = 18 combinations of RandomForestRegressor hyperparamter values, and it
# will train each model five times i.e., there will be 18 X 5 = 90 rounds of training!

grid_search.best_params_
# This gives out the best combination of parameters like this:


# In[79]:


# The evaluation scores are also available
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
# In this example, we obtain the best solution by setting the max_features hyperparameter to 6, and then
# n_estimators hyperparameter to 30. The RMSE score for this combination is 49889, which is slightly better than
# the score you go tearlier using the default hyperparameter values (which was 52927).
# For Randomized search see Readme.mdf


# In[86]:


# GridSearchCV can also reveal the relative importance of each attribute for making accurate predictions:
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# cat_one_attribs contains the subcategories of ocean_proximity separately.
# The purpose of zip() is to map the similar index of multiple containers so that they can be used just 
# using as single entity. 


# In[91]:


# Evaluate the final model on the test set; just set the predictors and the labels from your test run, run your
# full_pipeline to transform the data, and evaluate the final model on the test set.
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[92]:


# Congratulations for your first working model.


# In[ ]:




