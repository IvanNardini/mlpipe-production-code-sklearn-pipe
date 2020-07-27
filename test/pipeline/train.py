# Read data
import numpy as np
import pandas as pd

#Preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#Pipeline
from pipeline import pipeline

#Model
from sklearn.ensemble import RandomForestClassifier
from postprocessing import PostProcessing

#Utils
import logging
import time
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

# Read configuration
stream = open('config.yaml', 'r')
config = yaml.load(stream)

def train():
    # Read Data
    data = pd.read_csv(config['paths']['data_path'])
    #Train Pipeline
    Pipeline_fit = pipeline.fit(data)
    X = Pipeline_fit.transform(data)
    # SMOTE
    smote =  SMOTE(random_state=0)
    X, y = smote.fit_resample(X[config['features_selected']], X[config['target']])
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=0)
    model = RandomForestClassifier(max_depth=25, 
                                    min_samples_split=5, 
                                    n_estimators=300,
                                    random_state=9)
    model.fit(X_train, y_train)

    #Save Model
    PostProcessing.save(model, config['paths']['pipe_path'])

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')
    start = time.time()
    train()
    end = time.time()
    duration = end - start
    logging.info('Training process successfully completed!')
    logging.info('Time for training: {} seconds!'.format(round(duration, 5)))