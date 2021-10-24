import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib 

def convert_to_label(term):

	if(term <= 36):
		return ' 36 months'
	else:
		return ' 60 months'


def lambda_handler(event, context):

	slots = event['currentIntent']['slots']
	State = slots['State']
	Amount = slots['Amount']
	AnnualIncome = slots['AnnualIncome']
	NumDelinq = slots['NumDelinq']
	DTI = slots['DTI']
	Term = slots['Term']
	EmpLength = slots['EmpLength']
	HomeOwn = slots['HomeOwn']
	OpenAcc = 3
	int_rate = 13.25
	Term = convert_to_label(Term)

	random_forest = joblib.load('rf.pkl')
	lbl_dict = joblib.load('label_encoder.pkl')

	data_input = pd.DataFrame({'addr_state':[State],
                           'int_rate':[int_rate],
                           'loan_amnt':[Amount],
                           'annual_inc':[AnnualIncome],
                           'term':[Term],
                           'emp_length':[EmpLength],
                           'home_ownership':[HomeOwn],
                           'dti':[DTI],
                           'open_acc':[OpenAcc]})
	for col in lbl_dict.keys(): 
		data_input[col] = lbl_dict[col].transform(data_input[col])

	prediction = random_forest.predict(data_input)[0]

	if(prediction == 'Fully Paid'):

		return {
			"dialogAction": {
				"type": "Close",
				"fulfillmentState": "Fulfilled",
				"message": {
					"contentType": "PlainText",
					"content": "You will likely qualify for a loan"
				}
			}
		}
	else:

		return {
			"dialogAction": {
				"type": "Close",
				"fulfillmentState": "Fulfilled",
				"message": {
					"contentType": "PlainText",
					"content": "You will most likely not qualify for a loan"
				}
			}

		}
