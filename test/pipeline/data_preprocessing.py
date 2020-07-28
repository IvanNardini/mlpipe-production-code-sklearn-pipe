'''
data_preprocessing module contains machine learning object templates
'''
# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler

#Sklearn pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#Utils
import sys
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
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
            logging.error('The config file is corrupted in encoding_meta key!')
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

class Dumminizer(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with dummies.

    Parameters
    ----------
    columns_to_dummies: list, default=None
    dummies_meta : dict, default=None

    """

    def __init__(self, columns_to_dummies, dummies_meta):
        if not isinstance(columns_to_dummies, list) and not isinstance(encoding_meta, dict):
            logging.error('The config file is corrupted in either nominal_predictors or encoding_meta keys!')
            sys.exit(1)
        else:
            self.columns_to_dummies = columns_to_dummies
            self.dummies_meta = dummies_meta

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.columns_to_dummies:
            cat_names = sorted(self.dummies_meta[var])
            obs_cat_names = sorted(list(set(X[var].unique())))
            dummies = pd.get_dummies(X[var], prefix=var)
            X = pd.concat([X, dummies], axis=1)
            if obs_cat_names != cat_names: #exception: when label misses 
                cat_miss_labels = ["_".join([var, cat]) for cat in cat_names if cat not in obs_cat_names] #syntetic dummy
                for cat in cat_miss_labels:
                    X[cat] = 0 
            X = X.drop(var, 1)
        return X

class Scaler(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with scaled variables via MinMaxScaler sklearn class.

    Parameters
    ----------
    columns_to_scale: list, default=None

    """

    def __init__(self, columns_to_scale):
        if not isinstance(columns_to_scale, list):
            logging.error('The config file is corrupted in features key!')
            sys.exit(1)
        else:
            self.columns_to_scale = columns_to_scale
            self.scaler = MinMaxScaler()

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        scaler_fit = self.scaler.fit(X[self.columns_to_scale]) 
        X[self.columns_to_scale] = scaler_fit.transform(X[self.columns_to_scale])
        return X

class Feature_selector(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with selected features

    Parameters
    ----------
    features_selected: list, default=None

    """

    def __init__(self, features_selected):
        if not isinstance(features_selected, list):
            logging.error('The config file is corrupted in features_selected key!')
            sys.exit(1)
        else:
            self.features_selected = features_selected

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.features_selected]
        return X