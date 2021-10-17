import pandas as pd
import urllib
from urllib.error import HTTPError


def get_poster(dataframe,saving_folder='poster'):

	for imdb,url in df.itertuples(index=False):
	    http=[]
	    try:
	        urllib.request.urlretrieve(url,'./'+saving_folder+'/'+str(imdb)+'.jpg')
	    except:
	        pass

if __name__ == '__main__':

	df = pd.read_csv('MovieGenre.csv', encoding="ISO-8859-1", usecols=["imdbId", "Title", "Genre", "Poster"])
	data=df.copy()
	data.drop(['Title','Genre'],axis=1,inplace=True)
	saving_folder='poster'
	
	get_poster(data,saving_folder)