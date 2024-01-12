#!/usr/bin/env python
# coding: utf-8

# # Introduction </br>
# The following script shows how to the workflow of a ML-Algorithm (Light GBM Regression) on a dataset (DDA purchase and newsletter data)

# # Installation</br>
# Install [python 3.11](https://www.python.org/downloads/release/python-3110/) </br>
# Install matplotlib: pip3 install matplotlib</br>
# Install seaborn   : pip3 install seaborn </br>
# Install lightgbm  : pip3 install lightgbm </br>

# # Imports

# In[319]:


import ciso8601
import math
from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import norm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import cross_val_score, cross_val_predict
import os
import optunity
import optunity.metrics
os.listdir("data")


# # Data Analysis

# In[320]:


df_train   = pd.read_csv("./data/ml-learning-data-2.csv",   sep=';', na_values=['', '-'], parse_dates=['LastNewsletter', 'date'], dayfirst=True)
df_predict = pd.read_csv("./data/ml-prediction-data-2.csv", sep=';', na_values=['', '-'], parse_dates=['LastNewsletter', 'date'], dayfirst=True)
df_train['DaysSinceLastNewsletter'] = df_train['DaysSinceLastNewsletter'].str.replace(',', '.').astype(float)
df_predict['DaysSinceLastNewsletter'] = df_predict['DaysSinceLastNewsletter'].str.replace(',', '.').astype(float)

df_train['id'] = [i for i in range(len(df_train.index))]
df_predict['id'] = [i for i in range(len(df_predict.index))]

df_predict.to_csv("expected_results.csv", sep=';')


# In[321]:


df_train.sample(3)


# DaysUntilNextPurchase is nothing we know at the time we want to predict so we drop it.<br/>Also Last Newsletter is directly connected to DaysSinceLastNewsletter and date. Therefore it has to be dropped as well.

# In[322]:


df_train.drop(['DaysUntilNextPurchase', 'LastNewsletter'], axis=1, inplace=True)


# In[323]:


"Training dataset: {}".format(df_train.shape)


# 13.292 Lines and 10 Columns

# In[324]:


df_predict.sample(3)


# UnitRelativeDays is our prediction column and DaysUntilNextPurchase is nothing we know at the time we want to predict so we drop it.<br/>Also Last Newsletter is directly connected to DaysSinceLastNewsletter and date. Therefore it has to be dropped as well.

# In[325]:


df_predict.drop(['UnitRelativeDays', 'DaysUntilNextPurchase', 'LastNewsletter'], axis=1, inplace=True)


# In[326]:


"Prediction dataset: {}".format(df_predict.shape)


# Dataset has 8749 Lines and 10 Columns

# In[327]:


df_train.describe()


# df_predict.describe()

# LastProductOrder is always 1 and not included in the training dataset so we can drop it

# In[328]:


df_predict.drop(['LastProductOrder'], axis=1, inplace=True)


# Date is in date format but ML needs floats or bools so we convert it into a timestamp

# In[329]:


df_train['date'] = pd.to_datetime(df_train['date']).astype('int64') // 10**9
df_predict['date'] = pd.to_datetime(df_predict['date']).astype('int64') // 10**9


# In[330]:


print('''After edditing,
Training dataset has {} lines and {} columns and
Prediction dataset has {} lines and {} columns
'''.format(df_train.shape[0], df_train.shape[1], df_predict.shape[0], df_predict.shape[1]))


# In[331]:


sns.histplot(df_train['UnitRelativeDays'], kde=True)


# Looks like a negative exponential function -> looks normal

# # Data Engineering

# Bring the data into the correct format. Herefor also all NULL values have to be replaced.

# ### def fill_missing:
# Fills the NULL values of the given columns (cols) with the given value (val)

# In[332]:


def fill_missing(df, cols, val):
    for col in cols:
        df[col] = df[col].fillna(val)


# ### def fill_missing_with_mode:
# Fills the NULL values of the given columns (cols) with the most used value in the column

# In[333]:


def fill_missing_with_mode(df, cols):
    """ Fill with the mode """
    for col in cols:
        df[col] = df[col].fillna(df[col].mode()[0])


# ### def addlogs
# Calculate a logagithm function on the given dataframe (res). The logarithm function is on the base of 10 and adds a Constant (1.01) to make sure the input is not <=0, The result is stored in a new column: *NAME*_log

# In[334]:


def addlogs(res, cols):
    """ Log transform feature list"""
    m = res.shape[1]
    for c in cols:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[c])).values)   
        res.columns.values[m] = c + '_log'
        m += 1
    return res


# ## Feature Engeneering
# We could also add some calculated features that are no direct columns in the inputdata, but we did this here already before. DaysSinceLastNewsletter, DaysUntilNextPurchase, FirstProductOrder, LastNewsletter, UnitRelativeDays are all feature engeneered columns

# # Feature handling
# ## Logs

# Calculate the log for all simple integer features

# In[335]:


loglist = ['Units', 'Preis'] # + ['UnitRelativeDays']
df_train = addlogs(df_train, loglist)
#df_predict = addlogs(df_predict, loglist)


# Calculate the natural log for real valued numbers

# In[336]:


#df_train['Units'] = np.log1p(df_train['Units'])
#df_train['Preis'] = np.log1p(df_train['Preis'])


# ## Fill missing values 

# In[337]:


fill_missing(df_train, ['Customerid', 'date', 'Orderid', 'Produktkey_Orders'], "Null")
fill_missing(df_train, ['FirstProductOrder', 'Units'], 1)
#fill_missing(df_train, ['DaysSinceLastNewsletter'], 17)

fill_missing_with_mode(df_train, ['Preis', 'UnitRelativeDays', 'DaysSinceLastNewsletter'])


# ## Type conversions
# Check that every type is correct for handling the data.<br/>
# If there would be any categorical columns that consist of numeric values we should transform the type in the dataframe to string. We do not have this in our usecase. <br/>Example:

# In[338]:


#df_train['WineQuality'] = df_train['WineQuality'].apply(str)
#df_train['YearCreated'] = df_train['YearCreated'].astype(str)


# ## Remove Outliers
# Identify Outliers by useing interquartile range (IQR). IQR represents the interquartile range calculated by Q3(75th percentile of the dataset) minus Q1(25th percentile of the dataset) (Q3–Q1).

# ### Identify Outliers

# In[339]:


def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   return outliers


# In[340]:


outliers = find_outliers_IQR(df_train[['UnitRelativeDays', 'DaysSinceLastNewsletter', 'Units', 'Preis']])
print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers


# In[341]:


def set_min_max(df, col, range):
    min = df[col].mean() - range*df[col].std()
    max = df[col].mean() + range*df[col].std()
    df[col] = np.where(df[col] > max,
                max,
                   np.where(df[col] < min, min,
                            df[col]
                           )

                )

set_min_max(df_train, 'UnitRelativeDays', 3)
set_min_max(df_train, 'DaysSinceLastNewsletter', 3)
set_min_max(df_train, 'Units', 2)
set_min_max(df_train, 'Preis', 1)
df_train.describe()


# ## Handle Categoricals

# In[342]:


def fix_missing_cols(in_train, in_test):
    missing_cols = list(set(in_train.columns) - set(in_test.columns))
    
    # Create a DataFrame with missing columns and default value 0
    missing_cols_df = pd.DataFrame(0, index=in_test.index, columns=missing_cols)
    
    # Concatenate the missing columns DataFrame with in_test
    in_test = pd.concat([in_test, missing_cols_df], axis=1)
    
    # Ensure the order of columns in the test set is the same as in the train set
    in_test = in_test[in_train.columns]
    
    return in_test

def dummy_encode(in_df_train, in_df_test):
    df_train = in_df_train
    df_predict = in_df_test
    categorical_feats = [
        f for f in df_train.columns if df_train[f].dtype == 'object' and f != "id"
    ]
    print(categorical_feats)
    for f_ in categorical_feats:
        prefix = f_
        df_train = pd.concat([df_train, pd.get_dummies(df_train[f_], prefix=prefix)], axis=1).drop(f_, axis=1)
        df_predict = pd.concat([df_predict, pd.get_dummies(df_predict[f_], prefix=prefix)], axis=1).drop(f_, axis=1)
        df_predict = fix_missing_cols(df_train, df_predict)
    return df_train, df_predict


# In[343]:


#set(df_train.columns) - set(df_predict.columns)
df_train, df_predict = dummy_encode(df_train, df_predict)


# In[344]:


print("Shape train: %s, test: %s" % (df_train.shape, df_predict.shape))


# ## Feature Engeneering

# In[345]:


#df_train.corr(numeric_only=True)


# In[346]:


def load_poly_features(df_train, df_predict, cols):
    """
    USeful function to generate poly terms
    :param df_train: The training data frame
    :param df_predict: The test data frame
    :return: df_poly_features, df_poly_features_test - The training polynomial features + the test
    """
    print('Loading polynomial features..')
    # Make a new dataframe for polynomial features
    poly_features = df_train[cols]
    poly_features_test = df_predict[cols]

    # imputer for handling missing values
    imputer = Imputer(strategy='median')

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)
    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    print('Polynomial Features shape: %s' % str(poly_features.shape))
    
    # Generate column names for the DataFrame
    column_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
    
    df_poly_features = pd.DataFrame(poly_features, columns=column_names)
    df_poly_features_test = pd.DataFrame(poly_features_test, columns=column_names)
    df_poly_features['id'] = df_train['id']
    df_poly_features_test['id'] = df_predict['id']
    print('Loaded polynomial features')
    return df_poly_features, df_poly_features_test


# In[347]:


correlated_cols = ['Units', 'Preis']
df_train_poly, df_predict_poly =  load_poly_features(df_train, df_predict, cols=correlated_cols)
print("Shape train: %s, predict: %s" % (df_train_poly.shape, df_predict_poly.shape))


# In[348]:


df_train = df_train.merge(right=df_train_poly.reset_index(), how='left', on='id')
df_predict = df_predict.merge(right=df_predict_poly.reset_index(), how='left', on='id')


# In[349]:


print("Shape train: %s, predict: %s" % (df_train.shape, df_predict.shape))


# # Light GBM

# In[350]:


y = df_train["UnitRelativeDays"]
y.sample(3)


# In[351]:


print("Shape train: %s, test: %s" % (df_train.shape, df_predict.shape))


# In[352]:


df_train.sample(2)


# In[353]:


df_train.drop(["UnitRelativeDays"], axis=1, inplace=True)
# The fix missing cols above will have added the target column to the test data frame, so this is a workaround to remove it
df_predict.drop(["UnitRelativeDays"], axis=1, inplace=True) 


# In[354]:


X_train, X_test, y_train, y_test = train_test_split( df_train, y, test_size=0.2, random_state=42)


# Define the objective function to be minimized (in this case, the RMSE) by using optunity to optimize the hyperparameters

# In[355]:


# Define the objective function to be minimized (in this case, the RMSE)
@optunity.cross_validated(x=X_train, y=y_train, num_folds=5)
def lightgbm_mse(x_train, y_train, x_test, y_test, search_space):
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_leaves': int(search_space.num_leaves),
        'learning_rate': search_space.learning_rate,
        'feature_fraction': search_space.feature_fraction
    }

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

    bst = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], early_stopping_rounds=10)

    y_pred = bst.predict(x_test, num_iteration=bst.best_iteration)
    mse = mean_squared_error(y_test, y_pred)
    return mse


# Define the search range for hyperparameters

# Run the optimization using Optunity

# In[356]:


search_space = {
    'num_leaves': [32, 128],
    'learning_rate': [0.001, 0.1],
    'feature_fraction': [0.5, 1.0]
}


# In[357]:


optimal_params, _, _ = optunity.minimize(lightgbm_mse, num_evals=300, search_space=search_space)
optimal_params


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( df_train, y, test_size=0.2, random_state=42)


# Below are the hyper parameters, here a tool like [Opunity](https://optunity.readthedocs.io/en/latest/) would be nice to use

# In[ ]:


params = {
    'objective': 'regression',
    'metric': ['l1'],
    'boosting_type': 'gbdt',
    'num_leaves': 128,
    'learning_rate': 0.005,
    'feature_fraction': 0.9
}


# In[ ]:


final_train_data = lgb.Dataset(X_train, label=y_train)
final_test_data = lgb.Dataset(X_test, label=y_test, reference=final_train_data)
final_bst = lgb.train(optimal_params, final_train_data, 100000, valid_sets=[final_test_data])


# In[ ]:


y_pred = final_bst.predict(X_train, num_iteration=final_bst.best_iteration)


# In[ ]:


#print('The rmse of prediction is:', round(mean_squared_log_error(y_pred, y_train) ** 0.5, 5))
mse = mean_squared_error(y_train, y_pred)
rmse = round((mse) ** 0.5, 5)
print('The RMSE of prediction is:', rmse)


# In[ ]:


prediction = final_bst.predict(df_predict, num_iteration=final_bst.best_iteration)


# In[ ]:


prediction


# In[ ]:


df_predict["UnitRelativeDays"] = prediction


# In[ ]:


df_predict.to_csv("results.csv", columns=["id", "UnitRelativeDays"], index=False, sep=';')


# In[ ]:




