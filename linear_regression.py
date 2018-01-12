import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import ensemble
from sklearn.metrics import mean_squared_error


# 1. CRIM      per capita crime rate by town
# 2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS     proportion of non-retail business acres per town
# 4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. NOX       nitric oxides concentration (parts per 10 million)
# 6. RM        average number of rooms per dwelling
# 7. AGE       proportion of owner-occupied units built prior to 1940
# 8. DIS       weighted distances to five Boston employment centres
# 9. RAD       index of accessibility to radial highways
# 10. TAX      full-value property-tax rate per $10,000
# 11. PTRATIO  pupil-teacher ratio by town
# 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13. LSTAT    % lower status of the population
# 14. MEDV     Median value of owner-occupied homes in $1000's


# R^2 (coefficient of determination) regression score function.

# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

class RegressionModel:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def set_training_data(self, data_df, X_names, Y_name, n_train):
        self.X = data_df[X_names]
        # print "before"
        # print X
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        # print "after"
        # print X
        self.y = data_df[Y_name]
        self.X_train = self.X[:-n_train]
        self.X_test = self.X[-n_train:]
        self.y_train = self.y[:-n_train]
        self.y_test = self.y[-n_train:]

    def features_selection(type, X, y, **kargs):
        # X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
        # Variance is remove low-variance features which means that it remove features similar to constant features
        knn = KNeighborsClassifier(n_neighbors=4)
        if type == "VT":
            # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
            threshold = kargs.get('threshold', (.8 * (1 - .8)))
            sel = VarianceThreshold(threshold=threshold).fit_transform(X)
            return X
        elif type == "KBest":
            score_func = kargs.get('score_func', chi2)
            k = kargs.get('k', 2)
            # X = SelectKBest(chi2, k=2).fit_transform(X, y)
            X = SelectKBest(eval(score_func), k=k).fit_transform(X, y)
            return X
        elif type == "sfs":
            # sfs1 = SFS(knn,
            #            	k_features=3, forward=True,	floating=False,
            #            	verbose=2, scoring='accuracy', cv=0)
            score_func = kargs.get('score_func', knn)
            k_features = kargs.get('k_features', 3)
            floating = kargs.get('floating', False)
            verbose = kargs.get('verbose', 2)
            scoring = kargs.get('scoring', 'accuracy')
            cv = kargs.get('cv', 0)

            sfs = SFS(score_func,
                      k_features=k_features, forward=True, floating=floating,
                      verbose=verbose, scoring=scoring, cv=cv)
            sfs = sfs.fit(X, y)
            print '\nSequential Forward Selection ' + 'k=' + str(k_features)
            print sfs.k_feature_idx_
            print 'CV Score:'
            print sfs.k_score_
        elif type == "sbs":
            score_func = kargs.get('score_func', knn)
            k_features = kargs.get('k_features', 3)
            floating = kargs.get('floating', False)
            verbose = kargs.get('verbose', 2)
            scoring = kargs.get('scoring', 'accuracy')
            cv = kargs.get('cv', 0)

            sfs = SFS(score_func,
                      k_features=k_features, forward=False, floating=floating,
                      verbose=verbose, scoring=scoring, cv=cv)
            sfs = sfs.fit(X, y)
            print '\nSequential Backward Selection ' + 'k=' + str(k_features)
            print sfs.k_feature_idx_
            print 'CV Score:'
            print sfs.k_score_

    def linear_model(self):
        print "Linear Model"
        # Create linear regression object
        regr = LinearRegression()

        # Train the model using the training sets
        regr.fit(self.X_train, self.y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(self.X_test)

        # print "feature importance: ", regr.feature_importances_

        # Score
        model_score = regr.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print 'Score: \n', model_score
        # The coefficients
        print 'Coefficients: \n', regr.coef_
        # The mean squared error
        print "Mean squared error: %.2f" % mse
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2

        return ['linear', model_score, mse, r2]

    def lasso_model(self, alpha):
        print "Lasso Model"
        # alpha = 0.1
        regr = Lasso(alpha=alpha)

        y_pred = regr.fit(self.X_train, self.y_train).predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        # print lasso
        # print "r^2 on test data : %f" % r2
        print 'Score: \n', regr.score(self.X_test, self.y_test)
        # The coefficients
        print 'Coefficients: \n', regr.coef_
        # The mean squared error
        print "Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2_score(self.y_test, y_pred)
        model_score = regr.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return ['lasso', model_score, mse, r2]

    def elasticNet_model(self, alpha, l1_ratio):
        print "ElasticNet Model"
        # enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
        regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        y_pred = regr.fit(self.X_train, self.y_train).predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        # print enet
        # print "r^2 on test data : %f" % r2
        print 'Score: \n', regr.score(self.X_test, self.y_test)
        # The coefficients
        print 'Coefficients: \n', regr.coef_
        # The mean squared error
        print "Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2_score(self.y_test, y_pred)
        model_score = regr.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return ['elasticNet', model_score, mse, r2]

    def ridge_model(self, alpha):
        print "Ridge Model"
        # reg = Ridge(alpha = .5)
        regr = Ridge(alpha=alpha)
        regr.fit(self.X_train, self.y_train)
        # print "feature importance: ", reg.feature_importances_
        y_pred = regr.predict(self.X_test)
        # print reg.coef_
        r2 = r2_score(self.y_test, y_pred)
        # print "r^2 on test data : %f" % r2
        print 'Score: \n', regr.score(self.X_test, self.y_test)
        # The coefficients
        print 'Coefficients: \n', regr.coef_
        print 'Intercept: \n', regr.intercept_
        
        # The mean squared error
        print "Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2_score(self.y_test, y_pred)
        model_score = regr.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return ['ridge', model_score, mse, r2]

    def svr(self, type, **kargs):
        print "SVR "+type+" Model"
        # #############################################################################
        # Fit regression model
        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        # svr_lin = SVR(kernel='linear', C=1e3)
        # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        C = kargs.get('C', 1e3)
        if type == "rbf":
            gamma = kargs.get('gamma', 0.1)
            svr = SVR(kernel='rbf', C=C, gamma=gamma)
        elif type == "linear":
            svr = SVR(kernel='linear', C=C)
        elif type == "poly":
            degree = kargs.get('degree', 2)
            svr = SVR(kernel='poly', C=C, degree=degree)

        y_pred = svr.fit(self.X_train, self.y_train).predict(self.X_test)
        # print "feature importance: ", svr.feature_importances_
        r2 = r2_score(self.y_test, y_pred)

        # print "r^2 on test data : %f" % r2
        print 'Score: \n', svr.score(self.X_test, self.y_test)
        # The coefficients
        # print 'Coefficients: \n', svr.coef_
        # print 'Intercept: \n', svr.intercept_
        
        # The mean squared error
        print "Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2_score(self.y_test, y_pred)
        model_score = svr.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return ['SVR', model_score, mse, r2]

    def decisionTreeRegressor(self, max_depth):
        print "DecisionTreeRegressor Model"
        # Fit regression model
        regr = DecisionTreeRegressor(max_depth=max_depth)
        regr.fit(self.X_train, self.y_train)
        print "feature importance: ", regr.feature_importances_
        # Predict
        y_pred = regr.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)

        # print "r^2 on test data : %f" % r2
        print 'Score: \n', regr.score(self.X_test, self.y_test)
        # The coefficients
        # print 'Coefficients: \n', regr.coef_
        # print 'Intercept: \n', regr.intercept_
        
        # The mean squared error
        print "Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2_score(self.y_test, y_pred)
        model_score = regr.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return ['DecisionTreeRegressor', model_score, mse, r2]

    def randomForestRegressor(self, max_depth, random_state):
        print "RandomForestRegressor Model"
        regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        regr.fit(self.X_train, self.y_train)
        print "feature importance: ", regr.feature_importances_
        y_pred = regr.predict(self.X_test)
        # print y_pred
        r2 = r2_score(self.y_test, y_pred)
        # print "r^2 on test data : %f" % r2
        print 'Score: \n', regr.score(self.X_test, self.y_test)
        # The coefficients
        # print 'Coefficients: \n', regr.coef_
        # print 'Intercept: \n', regr.intercept_
        
        # The mean squared error
        print "Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2_score(self.y_test, y_pred)
        model_score = regr.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return ['RandomForestRegressor', model_score, mse, r2]

    def test(self):

        # rng = np.random.RandomState(0)

        # # #############################################################################
        # # Generate sample data
        # X = 5 * rng.rand(10000, 1)
        # y = np.sin(X).ravel()

        # # Add noise to targets
        # y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

        # X_plot = np.linspace(0, 5, 100000)[:, None]

        # # #############################################################################
        # # Fit regression model
        # train_size = 100
        train_size = len(self.X_train)
        svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                           param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                       "gamma": np.logspace(-2, 2, 5)})

        kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                          param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                      "gamma": np.logspace(-2, 2, 5)})

        t0 = time.time()
        svr.fit(self.X_train, self.y_train)
        svr_fit = time.time() - t0
        print("SVR complexity and bandwidth selected and model fitted in %.3f s"
              % svr_fit)

        t0 = time.time()
        kr.fit(self.X_train, self.y_train)
        kr_fit = time.time() - t0
        print("KRR complexity and bandwidth selected and model fitted in %.3f s"
              % kr_fit)

        sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
        print("Support vector ratio: %.3f" % sv_ratio)

        t0 = time.time()
        y_svr = svr.predict(self.X_test)
        svr_predict = time.time() - t0
        print("SVR prediction for %d inputs in %.3f s"
              % (self.X_test.shape[0], svr_predict))

        t0 = time.time()
        y_kr = kr.predict(self.X_test)
        kr_predict = time.time() - t0
        print("KRR prediction for %d inputs in %.3f s"
              % (self.X_test.shape[0], kr_predict))


        # #############################################################################
        # Look at the results
        sv_ind = svr.best_estimator_.support_
        plt.scatter(self.X[sv_ind], self.y[sv_ind], c='r', s=50, label='SVR support vectors',
                    zorder=2, edgecolors=(0, 0, 0))
        plt.scatter(self.X[:train_size], self.y[:train_size], c='k', label='data', zorder=1,
                    edgecolors=(0, 0, 0))
        plt.plot(self.X_test, y_svr, c='r',
                 label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
        plt.plot(self.X_test, y_kr, c='g',
                 label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
        plt.xlabel('data')
        plt.ylabel('target')
        plt.title('SVR versus Kernel Ridge')
        plt.legend()

        # Visualize training and prediction time
        plt.figure()

        # Generate sample data
        X = self.X
        y = self.y
        sizes = np.logspace(1, 4, 7, dtype=np.int)
        for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
                                                   gamma=10),
                                "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
            train_time = []
            test_time = []
            for train_test_size in sizes:
                t0 = time.time()
                estimator.fit(X[:train_test_size], self.y[:train_test_size])
                train_time.append(time.time() - t0)

                t0 = time.time()
                estimator.predict(self.X_test[:1000])
                test_time.append(time.time() - t0)

            plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
                     label="%s (train)" % name)
            plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
                     label="%s (test)" % name)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Train size")
        plt.ylabel("Time (seconds)")
        plt.title('Execution Time')
        plt.legend(loc="best")

        # Visualize learning curves
        plt.figure()

        svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
        kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
        train_sizes, train_scores_svr, test_scores_svr = \
            learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                           scoring="neg_mean_squared_error", cv=10)
        train_sizes_abs, train_scores_kr, test_scores_kr = \
            learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                           scoring="neg_mean_squared_error", cv=10)

        plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
                 label="SVR")
        plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",
                 label="KRR")
        plt.xlabel("Train size")
        plt.ylabel("Mean Squared Error")
        plt.title('Learning curves')
        plt.legend(loc="best")

        plt.show()

    def gradientBoostingRegressor(self, **kargs):
        # #############################################################################
        # Fit regression model
        # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
        #           'learning_rate': 0.01, 'loss': 'ls'}
        print "gradientBoostingRegressor Model"
        clf = ensemble.GradientBoostingRegressor(**kargs)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print 'Score: \n', clf.score(self.X_test, self.y_test)
        # The coefficients
        # print 'Coefficients: \n', regr.coef_
        # print 'Intercept: \n', regr.intercept_
        
        # The mean squared error
        print "Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        print 'Variance score: %.2f' % r2_score(self.y_test, y_pred)


        # mse = mean_squared_error(self.y_test, y_pred)
        # print("MSE: %.4f" % mse)

        # #############################################################################
        # Plot training deviance

        # compute test set deviance
        test_score = np.zeros((kargs['n_estimators'],), dtype=np.float64)

        for i, y_pred in enumerate(clf.staged_predict(self.X_test)):
            test_score[i] = clf.loss_(self.y_test, y_pred)

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

        # #############################################################################
        # Plot feature importance
        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        feature_names =['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        feature_names = np.asarray(feature_names)
        plt.yticks(pos, feature_names[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()
        model_score = clf.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return ['gradientBoostingRegressor', model_score, mse, r2]

if __name__ == '__main__':
    # call data
    data_file = "housing_data.csv"
    column_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                   'MEDV']
    raw_data = pd.read_csv(data_file, header=None)
    # print raw_data.columns
    raw_data.columns = column_list
    # print raw_data

    X_names = column_list[:-1]
    Y_name = column_list[-1]
    # print X_names
    # print Y_name
    train_ratio = 0.7
    n_train = int(len(raw_data) * train_ratio)
    test1 = RegressionModel()
    test1.set_training_data(raw_data, X_names, Y_name, n_train)

    value_lists = []

    print "=========================================================="
    
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    result_list = test1.gradientBoostingRegressor(**params)
    value_lists.append(result_list)
    print "=========================================================="
    # linear
    result_list = test1.linear_model()
    value_lists.append(result_list)
    
    print "=========================================================="
    max_depth_list = [3,5,7,9,10]
    # decisionTree
    for max_depth in max_depth_list:
        print "max depth", max_depth
        result_list = test1.decisionTreeRegressor(max_depth)
        value_lists.append(result_list)
    
        print "=========================================================="
    
    print "=========================================================="
    # decisionTree
    for max_depth in max_depth_list:
        print "max depth", max_depth
        result_list = test1.randomForestRegressor(max_depth,0)
        value_lists.append(result_list)
    
        print "=========================================================="
    
    print "=========================================================="
    
    alpha_list = [2.0,1.5,1.0, 0.5, 0.1, 0.05, 0.01]
    
    # elasticNet
    
    # alpha = 1.0
    l1_ratio_list = [0.5, 0.25, 0.1]
    l1_ratio = .5
    for alpha in alpha_list:   
        for l1_ratio in l1_ratio_list:
            print "alpha is", alpha, "l1_ratio is", l1_ratio 
            result_list = test1.elasticNet_model(alpha, l1_ratio)
            value_lists.append(result_list)
    
            print "=========================================================="
    
    print "=========================================================="
    # lasso
    # alpha = 0.1
    for alpha in alpha_list:
        print "alpha is", alpha
        result_list = test1.lasso_model(alpha)
        value_lists.append(result_list)
    
        print "=========================================================="
    
    print "=========================================================="
    # lasso
    # alpha = 0.5
    # alpha_list = [1.0, 0.5, 0.1, 0.05, 0.01]
    for alpha in alpha_list:
        print "alpha is", alpha
        result_list = test1.ridge_model(alpha)
        value_lists.append(result_list)
    
        print "=========================================================="
    
    print "=========================================================="
    gamma_list = [.1]
    for gamma in gamma_list:
        print "gamma is", gamma
        kargs = {'C':1e3, 'kernel':'rbf', 'gamma':gamma}
        result_list = test1.svr("rbf")
        value_lists.append(result_list)
    
        print "=========================================================="
    
    print "=========================================================="
    kargs = {'C':1e3, 'kernel':'linear'}
    result_list = test1.svr("linear")
    value_lists.append(result_list)
    
    
    print "=========================================================="
    degree_list = [3]
    for degree in degree_list:
        print "degree is", degree
        kargs = {'C':1e3, 'kernel':'poly', 'degree':degree}
        result_list = test1.svr("poly")
        value_lists.append(result_list)
    
        print "=========================================================="
    
    
    print "=========================================================="
    
    #######################
    df = pd.DataFrame(value_lists, columns=['model', 'score','mse','r2'])
    
    df = df.ix[df['r2'] >= 0.4] 
    df = df[['model','r2','mse']]
    # df = df.set_index("model")
    df = df.sort_values('mse')
    print df
    print test1.test()