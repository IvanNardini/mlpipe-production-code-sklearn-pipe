'''
Compile pipeline contains the pipeline object
'''
import data_preprocessing as Data_Prep
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

# Read configuration
stream = open('config.yaml', 'r')
config = yaml.load(stream)

pipeline = Pipeline(
    [
        ('Data_Preparer', Data_Prep.Data_Preparer(dropped_columns=config['dropped_columns'], renamed_columns=config['renamed_columns'])),

        ('Missing_Imputer', Data_Prep.Missing_Imputer(missing_predictors=config['missing_predictors'], 
                                                      replace='missing')), 

        ('Binner', Data_Prep.Binner(binning_meta=config['binning_meta'])), 

        ('Encoder', Data_Prep.Encoder(encoding_meta=config['encoding_meta'])),

        ('Dumminizer', Data_Prep.Dumminizer(columns_to_dummies=config['nominal_predictors'], dummies_meta=config['dummies_meta'])), 

        ('Scaler', Data_Prep.Scaler(columns_to_scale=config['features'])),

        ('SMOTE', SMOTE(random_state=0)), 

        ('RandomForestClassifier', RandomForestClassifier(max_depth=25, min_samples_split=5, n_estimators=300, random_state=8))

    ]
)