#Data
import pandas as pd

#Preprocessing
from sklearn.model_selection import train_test_split

#Evaluate Pipeline
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

def score(pipe_path, input_data):
    
    pipeline = PostProcessing.load(pipe_path)
    predictions = pipeline.predict(input_data)

    return pipeline, predictions
   
if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    # Read Data
    data = pd.read_csv(config['paths']['data_path'])
    
    # Encode target
    target = config['target']
    variables = [col for col in data.columns if col != target]
    target_labels = sorted(set(data[target]))
    target_labels_dic = {label: index for index, label in enumerate(target_labels, 0)}
    data[target] = data[target].map(target_labels_dic)
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(data[variables], data[target],
                                                        test_size=0.20,
                                                        random_state=1) 
    logging.info('Scoring process started!')
    start = time.time()
    pipeline, predictions = score(config['paths']['pipe_path'], X_train)
    end = time.time()
    duration = end - start
    logging.info('Scoring process successfully completed!')
    logging.info('Time for Scoring: {} seconds!'.format(round(duration, 5)))

    ##################################################################################

    #Evaluate
    print("*"*20)
    print("Model Predictions".center(20, '*'))
    print("*"*20)
    print()
    print('First 10 prediticions are: {}'.format(predictions[:10]))
    print()
  
    print("*"*20)
    print("Model Assessment".center(20, '*'))
    print("*"*20)
    PostProcessing.evaluate_classification(X_train, y_train, pipeline, predictions)
