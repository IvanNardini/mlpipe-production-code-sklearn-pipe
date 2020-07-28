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
    target_labels = set(data[config['target']])
    target_labels_dic = {label: index for index, label in enumerate(target_labels, 0)}
    print(target_labels_dic)
    data[config['target']] = data[config['target']].map(target_labels_dic)
    
    #Split data
    target = config['target']
    variables = [col for col in data.columns if col != target]
    X_train, X_test, y_train, y_test = train_test_split(data[variables], data[target],
                                                        test_size=0.1,
                                                        random_state=0)    
    #Train Pipeline
    Pipeline_Fit = pipeline.fit(X_train, y_train)

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