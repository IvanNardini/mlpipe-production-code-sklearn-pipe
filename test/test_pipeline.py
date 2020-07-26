import pandas as pd
from Pipeline.pipeline import *

# Read configuration
stream = open('./Pipeline/config.yaml', 'r')
config = yaml.load(stream)

data = pd.read_csv(config['paths']['data_path'])

# def test_pipeline_data_preparer():
#     pass

