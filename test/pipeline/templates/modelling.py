#Data
from sklearn.model_selection import train_test_split
#Model
from sklearn.ensemble import RandomForestClassifier
#Sklearn pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
            self.model = None
            self.test_size = 0.1
            self.random_state = 9
            self.max_depth = 25
            self.min_samples_split = 5
            self.n_estimators= 300

    def fit(self, X, y):
        self.model = RandomForestClassifier(max_depth=self.max_depth, 
                                    min_samples_split=self.min_samples_split, 
                                    n_estimators=self.n_estimators,
                                    random_state=self.random_state_model)
        return self

    def predict(self, X):
        return self.model.predict(X)