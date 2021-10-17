import pandas as pd
import numpy as np
import parser

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('display.max_rows', None)


"""
models
"""
def linear_reg(x_train,y_train,x_test,y_test):

	Linear_model = LinearRegression()
	Linear_model.fit(x_train, y_train)
	y_predict=Linear_model.predict(x_test)
	
	return abs(y_predict-y_test).mean()

def random_forest(x_train,y_train,x_test,y_test,n_estimators=100):

	regressor = RandomForestRegressor(n_estimators = n_estimators , random_state = 0) 
	regressor.fit(x_train,y_train)   
	RFR_pred = regressor.predict(x_test)
	
	return abs(RFR_pred-y_test).mean()


def random_forest_with_hyperparameter(x_train,y_train,x_test,y_test,n_estimators=[50,100]):

	from sklearn.model_selection import GridSearchCV
	clf =GridSearchCV(RandomForestRegressor(random_state = 0) ,{
	    'n_estimators':n_estimators,
	}, scoring='neg_mean_squared_error',cv=2,return_train_score=False)



	clf.fit(x_train,y_train)   
	RFR_pred_grid = clf.predict(x_test)
	abs(RFR_pred_grid-y_test).mean()

	
	return abs(RFR_pred-y_test).mean()

def XGBoost(x_train,y_train,x_test,y_test, eval_metric='mae',early_stopping_rounds=10 ):
	
	xgb_model=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
			colsample_bynode=1, colsample_bytree=1, eta=0.02,
			eval_metric='mae', gamma=0, importance_type='gain',
			learning_rate=0.05, max_delta_step=0, max_depth=10,
			min_child_weight=3, missing=None, n_estimators=200, n_jobs=1,
			nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,
			reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
			subsample=1, verbosity=1)

	eval_set = [(x_train, y_train), (x_test, y_test)]
	xgb_model.fit(x_train,y_train , eval_set=eval_set ,verbose=False , eval_metric=eval_metric ,early_stopping_rounds=early_stopping_rounds )

	y_predict=xgb_model.predict(x_test)

	return abs(y_predict-y_test).mean()

def XGBoost_hyperparameters_tuning(x_train,y_train,x_test,y_test,n_estimators=[100, 200], max_depth=[ 10, 20 ,30 , 40], eval_metric=['mae'] ):
	params={
		"learning_rate"    :[0.05],
		'eta'              :[0.02],
		"max_depth"        :max_depth,
		"n_estimators"     :n_estimators,
		"early_stopping_rounds":[10], 
		"eval_metric" : ["mae"], 
		"eval_set" : [[x_test, y_test]],
		'min_child_weight':[3]
	}

	classifier = XGBRegressor()

	xgb_clf=GridSearchCV(classifier, params, verbose=1,             
	         cv=TimeSeriesSplit(n_splits=5).get_n_splits([x_train, y_train]))


	xgb_clf.fit(x_train,y_train)

	y_predict_test =  xgb_clf.predict(x_test)

	return abs(y_predict_test-y_test).mean()



def handling_dtype(data):
	data['3G']=data['3G'].astype(str)
	data['4G']=data['4G'].astype(str)
	data['5G']=data['5G'].astype(str)
	data['VoLTE']=data['VoLTE'].astype(str)
	data['Wi-Fi']=data['Wi-Fi'].astype(str)
	data['IR Blaster']=data['IR Blaster'].astype(str)
	data['HDMI']=data['HDMI'].astype(str)
	data['Quad Sim']=data['Quad Sim'].astype(str)
	data['Triple Sim']=data['Triple Sim'].astype(str)
	data['Single Sim']=data['Single Sim'].astype(str)
	data['Dual Sim']=data['Dual Sim'].astype(str)
	data['NFC']=data['NFC'].astype(str)

	data['Front_cam_2'][data['Front_cam_2']=="[' 8  ', ' TOF 3D Dual Front Camera']"]=data['Front_cam_2'].mode()
	data['Front_cam_2']=data['Front_cam_2'].astype(float)

	data['Front_cam_1']=data['Front_cam_1'].astype(str)
	data['Front_cam_1']=data['Front_cam_1'].apply(lambda x:x.replace(' ','').replace("'",'').replace('[','').split(',')[0] if '[' in x else x )
	data['Front_cam_1']=data['Front_cam_1'].astype(float)
	data['Front_cam_1'].unique()

	data['Single_cam'][data['Single_cam']=='No']=data['Single_cam'].mode()
	data['Single_cam']=data['Single_cam'].astype(float)
	return data

def feature_engineering(data):
	data.dropna(subset=['score-val'],inplace=True)

	features=[feature for feature in data.columns]


	numeric_feature = [feature for feature in features if data[feature].dtypes!='O']


	categorical_features=[feature for feature in data.columns if data[feature].dtype =='O']
	categorical_features= list(set(categorical_features)-set(['pid']))


	for feature in numeric_feature:
		nan = data[feature].median()
		data[feature].fillna(nan, inplace=True)

	for feature in categorical_features:
		nan = data[feature].mode()[0]
		data[feature].fillna(nan, inplace=True)

	for feature in categorical_features:
	    labels_ordered=data.groupby([feature])['price'].mean().sort_values().index
	    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
	    data[feature]=data[feature].map(labels_ordered)


	data = data.sample(frac=1).reset_index(drop=True)
	return data

"""
feature scaling
"""
def feature_scaling(train_data):	
	feature_scale=[feature for feature in train_data.columns ]


	scaler=MinMaxScaler()
	scaler.fit(train_data[feature_scale])


	train_data = pd.DataFrame(scaler.transform(train_data[feature_scale]), columns=feature_scale)

	return train_data


def feature_scaling(x_train,y_train):
	# lasso
	feature_sel_model = SelectFromModel(Lasso(alpha=0.0001, random_state=0)) 
	feature_sel_model.fit(x_train, y_train)

	feature_sel_model.get_support()
	selected_feat = x_train.columns[(feature_sel_model.get_support())]

	return selected_feat



if __name__ == '__main__':


	parser=argparse.ArgumentParser()
	parser.add_argument('--model',help='model')


	model=parser.model

	df= pd.read_csv('modify.cvs')
	data=df.copy()

	data = handling_dtype(data)

	data = feature_engineering(data)

	"""
	data spillting into train ,test dataset.
	"""
	train_data = data[:1500]
	test_data  = data[1500:]

	orig_df=data[['pid','score-val']][1500:]
	train_data.drop(['Unnamed: 0','name','pid'],inplace=True,axis=1)

	#feature selection
	train_data = feature_sel(train_data)

	# creating labels and input
	train_data_f = train_data.drop(['score-val'],axis=1)
	Pricing = train_data['score-val']

	#spilting further into validation dataset and train dataset.
	x_train,x_test,y_train,y_test=train_test_split(train_data_f,Pricing,test_size=.05)


	selected_feat = feature_selection(x_train,y_train)

	x_train	=	x_train[selected_feat]
	x_test  =	x_test[selected_feat]


	# preparing test dataset for input
	test_data.drop(['Unnamed: 0','name','pid'],inplace=True,axis=1)
	test_data = pd.DataFrame(scaler.transform(test_data[feature_scale]), columns=feature_scale)
	test_data_f = test_data.drop(['score-val'],axis=1)
	test_data = test_data[selected_feat]

	test_data_f=test_data_f[selected_feat]

	print("linear model: mean absolute error"+linear_reg(x_train,y_train,x_test,y_test))
	print("Random forest model: mean absolute error"+random_forest(x_train,y_train,x_test,y_test))
	print("XGB model: mean absolute error"+XGBoost(x_train,y_train,x_test,y_test, eval_metric='mae',early_stopping_rounds=10 ))
	print("XGB model with hyperparameters tuning : mean absolute error"+XGBoost_hyperparameters_tuning(x_train,y_train,x_test,y_test,n_estimators=[100,200,300,400], max_depth=[ 40 , 60 ,70 , 80 ,90 ,100 ,110 ,200], eval_metric=['mae'] ))





	r=data['score-val'][:1500].max()-data['score-val'][:1500].min()
	

	y_predict = XGBoost_hyperparameters_tuning(x_train,y_train,x_test,y_test,n_estimators=[100, 200], max_depth=[ 10, 20 ,30 , 40], eval_metric=['mae'] )
	y_predict = y_predict*(r)+data['score-val'][1500:].min()

	
	orig_df['predict']=y_predict
	
	print("Error in SPEC score "+abs(orig_df['score-val']-orig_df['predict']).mean())
