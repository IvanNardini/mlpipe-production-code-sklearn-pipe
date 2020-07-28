# Read data
import pandas as pd

#Preprocessing
from sklearn.model_selection import train_test_split

#Pipeline
from pipeline import pipeline

#Model
from postprocessing import PostProcessing

#Utils
import logging
import time
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

# Read configuration
stream = open('config.yaml', 'r')
config = yaml.load(stream)

def train():
    # Read Data
    data = pd.read_csv(config['paths']['data_path'])
    
    # Encode target
    target = config['target']
    variables = [col for col in data.columns if col != target]
    target_labels = sorted(set(data[target]))
    target_labels_dic = {label: index for index, label in enumerate(target_labels, 0)}
    data[target] = data[target].map(target_labels_dic)
    
    #Split data
    # X_train, X_test, y_train, y_test = train_test_split(data[variables], data[target],
    #                                                     test_size=0.20,
    #                                                     random_state=1)    
    #Train Pipeline
    Pipeline_Fit = pipeline.fit(data[variables], data[target])

    #Save Model
    PostProcessing.save(Pipeline_Fit, config['paths']['pipe_path'])

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')
    start = time.time()
    train()
    end = time.time()
    duration = end - start
    logging.info('Training process successfully completed!')
    logging.info('Time for training: {} seconds!'.format(round(duration, 5)))