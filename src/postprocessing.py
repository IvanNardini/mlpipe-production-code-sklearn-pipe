from sklearn.metrics import classification_report
import joblib

class PostProcessing:
    @staticmethod
    def evaluate_classification(model, X_train, y_train, X_test, y_test):
        '''
        Evaluate classification
        params: model, X_train, y_train, X_test, y_test
        returns: None
        '''
        #Evaluate Train Sample
        # predictions_train = model.predict(X_train)
        # score_train = round(model.score(X_train, y_train), 2)
        # classification_train = classification_report(y_train, predictions_train)
        # print()
        # print('score: {}'.format(score_train))
        # print()
        # print('Classification report - Training')
        # print(classification_train)

        #Evaluate Test Sample
        predictions_test = model.predict(X_test)
        score_test = round(model.score(X_test, y_test), 2)
        classification_test = classification_report(y_test, predictions_test)
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
        
