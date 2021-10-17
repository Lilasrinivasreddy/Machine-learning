from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as ureq
import requests 
import pandas as pd

"""
Passing all 25 entries from single page , to get all feilds . 
"""
def get_details(containers , j):
	for i,container in enumerate(containers):

		try:
			val= container.div.div.text
			name  = container.h2.a.text

			feature_all=[]

			# getting 'pros' feature 
			product_features = container.find('ul',{"class":"pros"})
			product_features = product_features.find_all('span')


			for feature in product_features:
				feature_all.append(feature.text)

			# getting 'cons' feature 
			product_features = container.find('ul',{"class":"cons"})
			product_features = product_features.find_all('span')
			for feature in product_features:
				feature_all.append(feature.text)

			features  =feature_all

			extra = container.find('div',{'class':'extra'})
			price =extra.span.text



			rating = extra.div.span['title']
			pid = extra.find('div',{'class':"button compare"})['pid']

			# suming 'i' local iterator to 'j' final indexing of dataframe till last page 
			ii=i+j
			df.loc[ii]=[pid ,val , name, feature_all ,  price , rating ]

		except:
			pass

"""
Passing one by one url of the pages to get all the features

"""
def get_soup(my_url , i=0):

	page = requests.get(my_url)
	page_html = page.text
	page_soup = soup(page_html , 'html.parser')
	containers = page_soup.findAll("li",{"class":"f-mobiles"})
	get_details(containers, i)

if __name__ == '__main__':
	# creating dataframe 
	df = pd.DataFrame(columns=['pid','score-val','name','features','price','rating'])


	"""
	parsing diff value of page to extract the data

	"""
	for page in range(1,100):

		r= 'https://www.smartprix.com/mobiles/?page='+str(page)
		i=(page-1)*25
		get_soup(r , i)

	#saving	created dataframe
	df.to_csv('dataset.csv')