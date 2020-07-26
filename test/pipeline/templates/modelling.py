#Data
from sklearn.model_selection import train_test_split
#Model
from sklearn.ensemble import RandomForestClassifier
#Sklearn pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

class Classifier(ClassifierMixin, BaseEstimator):

    def __init__(self):
        self.rfor = None
        self.test_size = 0.1
        self.random_state_sample = 0
        self.random_state_model = 9
        self.max_depth = 25
        self.min_samples_split = 5
        self.n_estimators= 300


    def fit(self, X, y):
        X, y = X, y
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state_sample)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.rfor = RandomForestClassifier(max_depth=self.max_depth, 
                                    min_samples_split=self.min_samples_split, 
                                    n_estimators=self.n_estimators,
                                    random_state=self.random_state_model)
        self.rfor.fit(self.X_train, self.y_train)
        return self

    def predict(self, X):
        return self.rfor.predict(X)