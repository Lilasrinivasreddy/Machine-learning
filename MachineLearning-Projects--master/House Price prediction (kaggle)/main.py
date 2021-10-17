from feature_engineering import feature_engineering
from feature_selection import feature_selection
from Models import linear_model,xgb_model
import argparser
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('--train_dataset',help='address of train dataset')
	parser.add_argument('--test_dataset',help='address of test dataset')
	parser.add_argument('--model',help='model')

	train_dataset = parser.train_dataset
	test_dataset  = parser.test_dataset
	model=parser.model
	feature_engineering(train_dataset,test_dataset)
	feature_selection()


	if model=='linear':
		linear_model()
	if model=='xbg':
		xgb_model()

	elif:
		linear_model()
		xgb_model()

