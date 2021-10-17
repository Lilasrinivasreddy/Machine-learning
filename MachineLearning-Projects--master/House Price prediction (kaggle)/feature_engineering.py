import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def feature_engineering(train_dataset='train.csv',test_dataset='test.csv'):
	train=pd.read_csv(train_dataset)
	test=pd.read_csv(test_dataset)
	df1=train.copy()
	df2=test.copy()
	df2['SalePrice']=0
	df=pd.concat([df1,df2],axis=0)
	df.reset_index(drop=True,inplace=True)

	data=df.copy()
	categorical_features=[feature for feature in data.columns if data[feature].dtype =='O']

	categorical_features_nan=[feature for feature in categorical_features if data[feature].isnull().sum()>1 ]

	for feature in categorical_features_nan :
		data[feature].fillna('Missing',inplace=True)

	numeric_features = [feature for feature in data.columns if not data[feature].dtype =='O' ]
	temp_features=[feature for feature in numeric_features if 'yr' in feature.lower() or 'year' in feature.lower()]
	new_numeric_features=list(set(numeric_features)-set(temp_features+['Id']))


	new_numeric_features_nan = [feature for feature in new_numeric_features if data[feature].isnull().sum()>1]
	for feature in new_numeric_features_nan:
		nan_value=data[feature].median()
		data[feature+'nan']=np.where(data[feature].isnull(),1,0)
		data[feature].fillna(nan_value,inplace=True)


	for feature in temp_features:
		data[feature]=data['YrSold']-data[feature]

	for feature in temp_features:
		nan_value=df[feature].median()
		data[feature+'nan']=np.where(data[feature].isnull(),1,0)
		data[feature].fillna(nan_value,inplace=True)

	gauss_numeric_feature=[feature for feature in new_numeric_features if 0 not in data[feature].unique() and feature != 'SalePrice']

	for feature in gauss_numeric_feature:
		data[feature]=np.log(data[feature])


	ext_gauss_feature=list(set(new_numeric_features) - set(gauss_numeric_feature)-{'SalePrice'})

	for feature in ext_gauss_feature:
		data[feature]=data[feature].replace([0],1)
		data[feature]=np.log(data[feature])

	for feature in categorial_features:
		labels_ordered=data.groupby([feature])['LotFrontage'].mean().sort_values().index
		labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
		data[feature]=data[feature].map(labels_ordered)

	feature_scale=[feature for feature in data.columns if feature not in ['Id','SalePrice']]

	scaler=MinMaxScaler()
	scaler.fit(data[feature_scale])
	data = pd.concat([data[['Id','SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(data[feature_scale]), columns=feature_scale)],
                    axis=1)

	data.to_csv('X_train.csv',index=False)


if __name__ == '__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('--train_dataset',help='address of train dataset')
	parser.add_argument('--test_dataset',help='address of test dataset')

	train_dataset = parser.train_dataset
	test_dataset  = parser.test_dataset
	feature_engineering(train_dataset,test_dataset)