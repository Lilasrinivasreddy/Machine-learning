from flask import Flask,request,render_template
from model import get_popular,get_recommendations
from flask_sqlalchemy import SQLAlchemy
from get import get
import pandas as pd
app=Flask(__name__)

#app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///movei.db'
#db=SQLAlchemy(app)
inf2 = {2054, 648, 780, 588, 4306, 595, 788, 4886, 480, 34, 78499, 3114, 364, 1136, 1265, 1073, 2355, 1270, 1210}

def get_id(mov):
	df0=pd.read_csv('dataset/movies.csv')
	id=[]
	for movie in df0:
		movie=df0['movieId'][df0['title']==mov]
		#print(movie)
		id.append(movie[0])
	print(id)
	return id[0]



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
	if request.method == 'POST':
		global inf2
		mov_name=request.form['movie']
		rat=request.form['rating']
		#print(mov)
		mov=get_id(mov_name)

		mov2=zip([int(mov)],[int(rat)])
		f=get_recommendations(mov2)

		inf=get(f)

	
	return render_template('index.html',result=inf)



if __name__ == "__main__":
	app.run(debug=True)