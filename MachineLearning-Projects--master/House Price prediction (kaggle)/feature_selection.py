import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def feature_selection():
	df=pd.read_csv('X_train.csv')
	df1=pd.read_csv('train.csv')
	selected_feat_2=['Alley','PoolQC','Fence','MiscFeature']


	train_data=data[:len(df1)].drop(['SalePrice'],axis=1)
	test_data=data[len(df1):].drop(['SalePrice'],axis=1)


	salePrice=data['SalePrice'][:len(df1)]
	salePrice=pd.DataFrame(salePrice)

	x_train,x_test,y_train,y_test=train_test_split(train_data,salePrice,test_size=.05)
	x_train.fillna(1,inplace=True)


	feature_sel_model = SelectFromModel(Lasso(alpha=1, random_state=0)) 
	feature_sel_model.fit(x_train, y_train)


	selected_feat = x_train.columns[(feature_sel_model.get_support())]

	print('total features: {}'.format((x_train.shape[1])))
	print('selected features: {}'.format(len(selected_feat)))

	selected_feat=list(set(selected_feat)-set(selected_feat_2))

	x_train=x_train[selected_feat]
	x_test=x_test[selected_feat]

	x_train.to_csv('x_train.csv',index=False)
	x_test.to_csv('x_test.csv',index=False)
	y_train.to_csv('y_train.csv',index=False)
	y_test.to_csv('y_test.csv',index=False)
	test_data.to_csv('test.csv',index=False)

if __name__ == '__main__':
	feature_selection()