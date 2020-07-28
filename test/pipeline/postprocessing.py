from sklearn.metrics import classification_report
import joblib

class PostProcessing:
    @staticmethod
    def evaluate_classification(X, y, pipeline, predictions):
        '''
        Evaluate classification
        params: model, X_train, y_train, X_test, y_test
        returns: None
        '''

        #Evaluate Sample
        score_test = round(pipeline.score(X, y), 2)
        classification_test = classification_report(y, predictions)
        print()
        print('score: {}'.format(score_test))
        print()
        print('Classification report - Test')
        print(classification_test)
        return 0

    @staticmethod
    def save(model, path):
        '''
        Store model
        params: model, path
        return 0
        '''
        joblib.dump(model, path)
        return 0

    @staticmethod
    def load(path):
        '''
        Load model
        params: path
        return obj
        '''
        obj = joblib.load(path)
        return obj
        
