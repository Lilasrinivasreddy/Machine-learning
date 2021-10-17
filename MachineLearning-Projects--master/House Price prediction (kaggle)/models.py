import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score as cvs
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from math import sqrt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def linear_model(x_train ='x_train.csv',y_train ='y_train.csv',x_test ='x_test.csv',y_test ='y_test.csv'):


	Linear_model = LinearRegression()
	Linear_model.fit(x_train, y_train)
	scores=cvs(Linear_model,x_train,y_train,cv=10)
	print("accuracy of linearRegressor "+str(scores.mean()))

	rms = np.sqrt(np.square(np.asarray(np.log(y_predict)-np.log(y_test))).sum()/float(len(y_predict)))
	print('RMSE = {}'.format(rms))

	y_predict=Linear_model.predict(x_test)

	return y_predict

def xgb_model(x_train ='x_train.csv',y_train ='y_train.csv',x_test ='x_test.csv',y_test ='y_test.csv'):


	xgb_model=XGBRegressor()
	xgb_model.fit(x_train,y_train)
	print('Accuracy of Xgb : {}'.format(xgb_model.score(x_test,y_test)))

	y_predict=xgb_model.predict(x_test)
	rms = np.sqrt(np.square(np.asarray(np.log(y_predict)-np.log(y_test))).sum()/float(len(y_predict)))

	print('RMSE = {}'.format(rms))

	return y_predict

if __name__ == '__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('--x_train',help="address of x_train.csv")
	parser.add_argument('--y_train',help="address of x_train.csv")
	parser.add_argument('--x_test',help="address of x_test.csv")
	parser.add_argument('--y_train',help="address of y_test.csv")

	x_train=parser.x_train
	y_train=parser.y_train
	x_test=parser.x_test
	y_train=parser.y_test

	linear_model(x_train,y_train,x_test,y_test)
	xgb_model(x_train,y_train,x_test,y_test)

