import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

class HousePrice():
    def __init__(self, filename, column_list, y_name, feature_importance_list, train_ratio):
        data = pd.read_csv(filename, header=None)
        data = data.dropna()
        data.columns = column_list
        # column_list.remove('MEDV')
        self.column_list = feature_importance_list
        # column_list = feature_importance_list
        self.X = data[feature_importance_list]
        self.y = data[y_name]
        self.X, self.y = shuffle(self.X, self.y, random_state=13)
        self.X = self.X.astype(np.float32)
        offset = int(self.X.shape[0] * train_ratio)
        self.X_train, self.y_train = self.X[:offset], self.y[:offset]
        self.X_test, self.y_test = self.X[offset:], self.y[offset:]

    def grid_search(self, cv, **params):
        # To find best parameter for GradientBoostingRegressor, do GridSearch with parameters' dictionary
        gs = GridSearchCV(ensemble.GradientBoostingRegressor(min_samples_split=2, loss='ls'), params, cv=cv)
        gs.fit(self.X_train, self.y_train)
        print "**************************************"
        print "GradientBoost BEST PARAMS:", gs.best_params_

    def run_model(self, **params):
        # using GradientBoostingRegressor to predict continuous value as Y
        clf = ensemble.GradientBoostingRegressor(**params)
        # training
        clf.fit(self.X_train, self.y_train)
        # calculate Mean Squared Error for checking error between real Y and predicted Y
        mse = mean_squared_error(self.y_test, clf.predict(self.X_test))
        # To check underfitting and overfitting, do cross validation
        cv_score = cross_val_score(clf, self.X, self.y)
        print "Cross Val Score: ", cv_score
        print "MSE: %.4f" % mse
        return clf

    def predict(self, clf):
        # predict Y with clf trained model
        y_pred = clf.predict(self.X_test)
        # calculate r2 score to check coefficient between real Y and predicted Y
        score = r2_score(self.y_test, y_pred)
        print "R2 score: ", score

    def fit_func(self, **params):
        # start run_model for making model
        clf = self.run_model(**params)

        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
        for i, y_pred in enumerate(clf.staged_predict(self.X_test)):
            test_score[i] = clf.loss_(self.y_test, y_pred)

        # draw plot which shows the scores by increasing n_estimators
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')

        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        # plot bar for showing feature importance with sorted index
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(self.column_list)[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()

if __name__ == "__main__":
    # Load data from file
    data_file = "housing_data.csv"

    # set column name for raw data
    column_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                   'MEDV']
    y_name = 'MEDV'
    # set training data ratio in whole data
    train_ratio = 0.9
    feature_importance_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    # feature_importance_list = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'INDUS', 'RAD','CHAS']
    house = HousePrice(data_file, column_list, y_name, feature_importance_list, train_ratio)

    # grid search with gradientboostregression
    # For cross validation
    cv = 3
    # set parameters for grid search
    grid_params = {'n_estimators': [100, 150, 200, 500], 'max_depth': [3, 5, 7, 10], 'learning_rate': [0.01, 0.001]}
    house.grid_search(cv, **grid_params)

    # Make house price prediction model with optimal parameter from simulation
    # Retest with best_parameter_
    params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    house.fit_func(**params)

    params = {'n_estimators': 150, 'max_depth': 3, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = house.run_model(**params)
    house.predict(clf)

