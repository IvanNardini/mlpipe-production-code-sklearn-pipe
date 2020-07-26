'''
data_preprocessing module contains machine learning object templates
'''
# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Sklearn pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#Utils
import sys
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

class Data_Preparer(BaseEstimator, TransformerMixin):
    '''
    Drop and Rename columns
    :params: data, columns_to_drop
    :return: DataFrame
    '''
    
    def __init__(self, dropped_columns=None, renamed_columns=None):
        if not isinstance(dropped_columns, list) and not isinstance(renamed_columns, dict):
            logging.error('The config file is corrupted in data preparer columns!')
            sys.exit(1)
        else:
            self.variables = variables

    # We have fit method cause Sklearn 
    def fit(self, X, y=None):
        return self

    def transform(self, X, dropped_columns, renamed_columns):
        X = X.copy()
        X.drop(dropped_columns, axis=1, inplace=True)
        X.rename(columns=renamed_columns, inplace=True)

        return X

    # def Missing_Imputer(self, data, missing_predictors, replace='missing'):
    #     '''
    #     Imputes '?' character with 'missing' label
    #     :params: data, missing_predictors, replace
    #     :return: DataFrame
    #     '''
    #     data = data.copy()
    #     for var in missing_predictors:
    #         data[var] = data[var].replace('?', replace)
    #     return data

    # def Binner(self, data, binning_meta):
    #     '''
    #     Create bins based on variable distributions
    #     :params: data, var, new_var_name, bins, bins_labels
    #     :return: DataFrame
    #     '''
    #     data = data.copy()
    #     for var, meta in binning_meta.items():
    #         data[meta['var_name']] = pd.cut(data[var], bins = meta['bins'], labels=meta['bins_labels'], include_lowest= True)
    #         data.drop(var, axis=1, inplace=True)
    #     return data

    # def Encoder(self, data, encoding_meta):
    #     '''
    #     Encode all variables for training
    #     :params: data, var, mapping
    #     :return: DataFrame
    #     '''
    #     data = data.copy()
    #     for var, meta in encoding_meta.items():
    #         if var not in data.columns.values.tolist():
    #             pass
    #         data[var] = data[var].map(meta)
    #     return data

    # def Dumminizer(self, data, columns_to_dummies, dummies_meta):
    #     '''
    #     Generate dummies for nominal variables
    #     :params: data, columns_to_dummies, dummies_meta
    #     :return: DataFrame
    #     '''
    #     data = data.copy()
    #     for var in columns_to_dummies:
    #         cat_names = sorted(dummies_meta[var])
    #         obs_cat_names = sorted(list(set(data[var].unique())))
    #         dummies = pd.get_dummies(data[var], prefix=var)
    #         data = pd.concat([data, dummies], axis=1)
    #         if obs_cat_names != cat_names: #exception: when label misses 
    #             cat_miss_labels = ["_".join([var, cat]) for cat in cat_names if cat not in obs_cat_names] #syntetic dummy
    #             for cat in cat_miss_labels:
    #                 data[cat] = 0 
    #         data = data.drop(var, 1)
    #     return data

    # def Scaler(self, data, columns_to_scale):
    #     '''
    #     Scale variables
    #     :params:  data, columns_to_scale
    #     :return: DataFrame
    #     '''
    #     data = data.copy()
    #     scaler = MinMaxScaler()
    #     scaler.fit(data[columns_to_scale])
    #     data[columns_to_scale] = scaler.transform(data[columns_to_scale])
    #     return data

    # def Balancer(self, data, features_selected, target, random_state):
    #     '''
    #     Produce Syntetic sample with SMOTE
    #     :params: features_selected, target, random_state
    #     :return: X, y
    #     '''
    #     data = data.copy()
    #     smote = SMOTE(random_state=random_state)
    #     X, y = smote.fit_resample(data[features_selected], data[target])
    #     return X, y
        
    # def Data_Splitter(self, X, y, test_size, random_state):
    #     '''
    #     Split data in train and test samples
    #     :params: X, y
    #     :return: X_train, X_test, y_train, y_test
    #     '''
        
    #     X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                         y,
    #                                                         test_size=test_size,
    #                                                         random_state=random_state)
    #     return X_train, X_test, y_train, y_test
    
