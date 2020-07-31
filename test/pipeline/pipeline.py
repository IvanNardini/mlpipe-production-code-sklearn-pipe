'''
Compile pipeline contains the pipeline object
'''
import data_preprocessing as Data_Prep
import feature_engineering as Feat_Eng
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

DATA_INGESTION = config['data_ingestion']
PREPROCESSING = config['preprocessing']
FEATURES_ENGINEERING = config['features_engineering']
MODEL_TRAINING = config['model_training']

pipeline = Pipeline(
    [
        ('Dropper', Data_Prep.Dropper(dropped_columns=PREPROCESSING['dropped_columns'])),

        ('Renamer', Data_Prep.Renamer(renamed_columns=PREPROCESSING['renamed_columns'])),

        ('Missing_Imputer', Data_Prep.Missing_Imputer(missing_predictors=PREPROCESSING['missing_predictors'], 
                                                    replace='missing')), 

        ('Binner', Feat_Eng.Binner(binning_meta=FEATURES_ENGINEERING['binning_meta'])), 

        ('Encoder', Feat_Eng.Encoder(encoding_meta=FEATURES_ENGINEERING['encoding_meta'])),

        ('Dumminizer', Feat_Eng.Dumminizer(columns_to_dummies=FEATURES_ENGINEERING['nominal_predictors'])), 

        ('Scaler', Feat_Eng.Scaler(features=FEATURES_ENGINEERING['features'])),

        ('Feature_selector', Feat_Eng.Feature_selector(features_selected=FEATURES_ENGINEERING['features_selected'])), 

        ('SMOTE', SMOTE(random_state=9)), 

        ('RandomForestClassifier', RandomForestClassifier(max_depth=25, min_samples_split=5, n_estimators=300, random_state=8))

    ]
)