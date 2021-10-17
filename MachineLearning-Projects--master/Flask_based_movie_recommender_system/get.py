import pandas as pd
df1=pd.read_csv('dataset/links.csv')
df2=pd.read_csv('dataset/movies.csv')
#print(df2.head())
df = pd.merge(df1,df2,on="movieId")
#print(df.head())
def get(f):
	inf=[]
	#print(f)
	for movieid in f:
		
		movie_imdb=df['imdbId'][df['movieId']==int(movieid)]
		
		movie_imdb=movie_imdb.tolist()
		
		for a in movie_imdb:
			movie_imdb=a
			movie_imdb="static/poster/"+str(movie_imdb)+".jpg"

		movie_title=df['title'][df['movieId']==int(movieid)]
		movie_title=movie_title.tolist()
		for a in movie_title:
			movie_title=str(a)

		movie=[movie_title,movie_imdb]
		inf.append(movie)
	return inf



