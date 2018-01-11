import pandas as pd 
import mlxtend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


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



class RegressionModel:
	def __init__(self):
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None

	def set_training_data(self, data_df, X_names, Y_name, n_train):
		X = data_df[X_names]
		print "before"
		print X
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
		print "after"
		print X
		y = data_df[Y_name]
		self.X_train = X[:-n_train]
		self.X_test = X[-n_train:]
		self.y_train = y[:-n_train]
		self.y_test = y[-n_train:]

	def features_selection(type, X, y, **kargs)
		# X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
		# Variance is remove low-variance features which means that it remove features similar to constant features
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
			# knn = KNeighborsClassifier(n_neighbors=4)
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
			print '\nSequential Forward Selection '+'k='k_features
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
			print '\nSequential Backward Selection '+'k='k_features
			print sfs.k_feature_idx_
			print 'CV Score:'
			print sfs.k_score_

	def linear_model(self):
		# Create linear regression object
		self.regr = LinearRegression()

		# Train the model using the training sets
		self.regr.fit(self.X_train, self.y_train)

		# Make predictions using the testing set
		self.y_pred = self.regr.predict(self.X_test)

		# Score
		print 'Score: \n', self.regr.score(self.X_test, self.y_test)
		# The coefficients
		print 'Coefficients: \n', self.regr.coef_
		# The mean squared error
		print "Mean squared error: %.2f" % mean_squared_error(self.y_test, self.y_pred)
		# Explained variance score: 1 is perfect prediction
		print 'Variance score: %.2f' % r2_score(self.y_test, self.y_pred)
		# print "Linear model:", self.pretty_print_linear(self.regr.coef_)
		# print self.y_pred

if __name__ == '__main__':
	# call data 
	data_file = "housing_data.csv"
	column_list = ['CRIM', 'ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	raw_data = pd.read_csv(data_file, header=None)
	print raw_data.columns
	raw_data.columns = column_list
	print raw_data

	X_names = column_list[:-1]
	Y_name = column_list[-1]
	print X_names	
	print Y_name
	train_ratio = 0.7
	n_train = int(len(raw_data)*train_ratio)
	test1 = RegressionModel()
	test1.set_training_data(raw_data, X_names, Y_name, n_train)
	test1.linear_model()
