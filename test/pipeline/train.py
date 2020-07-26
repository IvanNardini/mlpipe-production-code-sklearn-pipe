# Read data
import numpy as np
import pandas as pd

#Pipeline
from templates.postprocessing import PostProcessing
from pipeline import model_pipeline

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
    # read data
    Data = pd.read_csv(config['paths']['data_path'])
    #Train Pipeline
    Pipeline = model_pipeline.fit(Data)
    print(Pipeline.transform(Data)['collision_type'])
    #Save Model
    PostProcessing.save(Pipeline, config['paths']['pipe_path'])

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')
    start = time.time()
    train()
    end = time.time()
    duration = end - start
    logging.info('Training process successfully completed!')
    logging.info('Time for training: {} seconds!'.format(round(duration, 5)))