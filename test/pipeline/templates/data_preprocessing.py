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

    """ A transformer that returns the transformer dataframe.

    Parameters
    ----------
    dropped_columns : list, default=None
    renamed_columns : dict, default=None

    """
    
    def __init__(self, dropped_columns=None, renamed_columns=None):
        if not isinstance(dropped_columns, list) and not isinstance(renamed_columns, dict):
            logging.error('The config file is corrupted either dropped_columns or renamed_columns keys!')
            sys.exit(1)
        else:
            self.dropped_columns = dropped_columns
            self.renamed_columns = renamed_columns

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        X.drop(self.dropped_columns, axis=1, inplace=True)
        X.rename(columns=self.renamed_columns, inplace=True)
        return X

class Missing_Imputer(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame
    with missings imputed.

    Parameters
    ----------
    missing_predictors : list, default=None
    replace : str, default=missing

    """

    def __init__(self, missing_predictors=None, replace='missing'):
        if not isinstance(missing_predictors, list):
            logging.error('The config file is corrupted in missing_predictors key!')
            sys.exit(1)
        else:
            self.missing_predictors = missing_predictors
            self.replace = replace

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.missing_predictors:
            X[var] = X[var].replace('?', self.replace)
        return X

class Binner(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with bins based on variable distributions.

    Parameters
    ----------
    binning_meta : list, default=None

    """

    def __init__(self, binning_meta=None):
        if not isinstance(binning_meta, dict):
            logging.error('The config file is corrupted in binning_meta key!')
            sys.exit(1)
        else:
            self.binning_meta = binning_meta

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var, meta in self.binning_meta.items():
            X[meta['var_name']] = pd.cut(X[var], bins = meta['bins'], labels = meta['bins_labels'], include_lowest = True)
            X.drop(var, axis=1, inplace=True)
        return X

class Encoder(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with variable encoded.

    Parameters
    ----------
    encoding_meta : list, default=None

    """
    
    def __init__(self, encoding_meta=None):
        if not isinstance(encoding_meta, dict):
            logging.error('The config file is corrupted in binning_meta key!')
            sys.exit(1)
        else:
            self.encoding_meta = encoding_meta


    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var, meta in self.encoding_meta.items():
            if var not in X.columns.values.tolist():
                pass
            X[var] = X[var].map(meta)
        return X


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
    
