import pandas as pd
import numpy as np
import argparse
import ast
"""
cleaning of scraped data
"""
def data_cleaning(dataset="dataset.csv",cleaned_dataset="cleaned_dataset.csv"):
	# reading scrap data

	df=pd.read_csv(dataset)

	# removing uncessory column 
	df.drop(['Unnamed: 0'],axis=1,inplace=True)

	data=df.copy()
	data.reset_index(drop=True,inplace=True)

	# first part of the name repesent 'brand'
	data['Brand']=data['name'].apply(lambda x: x.split()[0])


	data['price']=data['price'].apply(lambda x:x.replace('₹','').replace(',',''))

	# Not decided to seperate the all netwroking config as most of dataset have pretty common networking configuration 
	data['Networking']=data['features'].apply(lambda x:x.split("'")[1])
	data['rating']=data['rating'].apply(lambda x:x.replace('User ratings: ','').split('/')[0])

	# Seperating out different list element of feature which contains all the feature data.
	data['feature_2']=data['features'].apply(lambda x:x.split("'")[3].replace('u2009',' '))
	data['feature_3']=data['features'].apply(lambda x:x.split("'")[5].replace('u2009',' '))
	data['feature_4']=data['features'].apply(lambda x:x.split("'")[7].replace('u2009',' '))
	data['feature_5']=data['features'].apply(lambda x: x.split("'")[9].replace('u2009',' ') if (len(x.split("'"))>9)  else False )
	data['feature_6']=data['features'].apply(lambda x: x.split("'")[11].replace('u2009',' ') if (len(x.split("'"))>11)  else False )
	data['feature_7']=data['features'].apply(lambda x: x.split("'")[13].replace('u2009',' ') if (len(x.split("'"))>13)  else False )
	data['feature_8']=data['features'].apply(lambda x: x.split("'")[15].replace('u2009',' ') if (len(x.split("'"))>15)  else False )

	# Extracting diffent features which orginally embedded within a single lists.
	data['Processor(GHz)']=data['feature_2'].apply(lambda x:x.split(',')[2].split('\\')[0].replace(' ','') if len(x.split(','))==3 else np.nan)
	data['Processor']=data['feature_2'].apply(lambda x:x.split(',')[0] if len(x.split(','))==3 else np.nan)
	data['Core']=data['feature_2'].apply(lambda x:x.split(',')[1] if len(x.split(','))==3 else np.nan)
	data['RAM(GB)']=data['feature_3'].apply(lambda x:x.split(',')[0].replace('\\','').split()[0])
	data['Memory(GB)']=data['feature_3'].apply(lambda x:x.split(',')[1].replace('\\','').split()[0] if len(x.split(','))>1 else np.nan)
	data['Battery']=data['feature_4'].apply(lambda x:x.split(',')[0].split('\\')[0] )
	data['Screen_size(inches)']=data['feature_5'].apply( lambda x:x.split(',')[0].split(' ')[0])
	data['Display']=data['feature_5'].apply(lambda x:x.split(',')[1].replace('\\','') if len(x.split(','))==2 else np.nan)
	data['Rear_cam']=data['feature_6'].apply(lambda x:x.split('&')[0].replace('\\','').replace('MP','').split('+') if x!=False else np.nan)
	data['Front_cam']=data['feature_6'].apply(lambda x:x.split('&')[1].replace('\\','').replace('MP','').split('+') if x!=False and len(x.split('&'))==2 else np.nan)
	data['Memory_card_supported']=data['feature_7'].apply(lambda x:x.split('\\')[0].split(' ')[-1] if x!=False else np.nan)
	data['Android-v']=data['feature_8']

	# droping  lists which contain multiple features after getting all the features out of it.
	data.drop(['features','feature_2','feature_3','feature_4'	,'feature_5','feature_6','feature_7','feature_8'],axis=1,inplace=True)
	data['rating']=data['rating'].apply( lambda x:x if len(x)<6 else np.nan)

	
	# converting some features to correct datatype and handling some unwanted values 


	# handling price values in lakhs 
	data['price']=data['price'].apply(lambda x:float(x.split(' ')[0])*100000 if x.isnumeric()==False else x)
	data['price']=pd.to_numeric(data['price']) 


	# checking whether the values are numeric as battery capacity always be as '5100' (Mh already stripped), if not then putting 'null' there
	data['Battery']=data['Battery'].apply(lambda x:np.nan if x.isnumeric()==False else x)
	data['Battery']=pd.to_numeric(data['Battery'])

	# checking whether the values are numeric as memory card always be as '4' ('GB' already stripped), if not then putting 'null' there
	data['Memory_card_supported']=data['Memory_card_supported'].apply(lambda x:np.nan if str(x).isnumeric()==False else x)
	data['Memory_card_supported']=pd.to_numeric(data['Memory_card_supported'])


	# checking whether the values are numeric as RAM storage always be as '4' ('GB' already stripped), if not then putting 'null' there
	data['RAM(GB)']=data['RAM(GB)'].apply(lambda x:np.nan if x.isnumeric()==False else x)
	data['RAM(GB)']=pd.to_numeric(data['RAM(GB)'])


	# checking whether the values are float as screensize always be as '5.5' ('inches' already stripped), if not then putting 'null' there
	data['Screen_size(inches)']=data['Screen_size(inches)'].apply(lambda x:x.replace("\\",'') if isFloat(x)==False else x)
	data['Screen_size(inches)']=data['Screen_size(inches)'].apply(lambda x:np.nan if isFloat(x)==False else x)
	data['Screen_size(inches)']=pd.to_numeric(data['Screen_size(inches)'])


	# removing some wrong feed by comparing with max limit of particular feature
	data['RAM(GB)']=data['RAM(GB)'].apply(lambda x:x if x<=64 else np.nan)
	data['Screen_size(inches)']=data['Screen_size(inches)'].apply(lambda x:x if x<12 else np.nan)
	data['Processor(GHz)']=data['Processor(GHz)'].apply(lambda x:x if x<5 else np.nan)

	data['Processor_brand']=data['Processor'].apply(lambda x:x.split()[0] if type(x)==str  else x)

	data['Front_cam_1']=data['Front_cam'].apply(lambda x:front_camera_1(x))

	data['Front_cam_2']=data['Front_cam'].apply(lambda x:front_camera_2(x))

	categories=[]
	for category in data['Networking'].unique():
		category=category.split(',')
		for cat in category:
    		categories.append(cat)
	categories=list(dict.fromkeys(categories))

	data['Dual Sim']=0
	data['3G']=0
	data['4G']=0
	data['VoLTE']=0
	data['Wi-Fi']=0
	data['IR Blaster']=0
	data['Single Sim']=0
	data['5G']=0
	data['NFC']=0
	data['HDMI']=0
	data['Wi-Fi']=0
	data['Quad Sim']=0
	data['Triple Sim']=0

	for i,net in enumerate(data['Networking']):
		networking_feature_creation(net,i)


	data['Quad_cam']  =0
	data['Single_cam']=0
	data['Triple_cam']=0
	data['Dual_cam']=0
	data['Penta_cam']=0

	data['Rear_cam'].fillna(data['Rear_cam'].mode()[0],inplace=True)
	for i,ele in enumerate(data['Rear_cam']):
		rear_cam(ele,i)

	data['Single_cam'][data['Single_cam']=='No']=13


	data['Android-v']=data['Android-v'].apply(lambda x:check_os(x))

	data.drop(['Rear_cam','Front_cam','Networking'],axis=1,inplace=True)

	handling_dtype(data)
	
	to_csv('latest.csv')

"""
Coverting all data into suitable data type format so we don't have to do it there.
"""
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

"""
checking OS version from android version features so we will able to classify them into less number of category.
"""

def check_os(x):
	x_list =x.split()
	os_list=['Android','OS','iOS']
	result=list(set(os_list)-(set(os_list)-set(x_list)))
	if len(result)==1:
		return x
	return np.nan


"""
Extracting value of 1st front camera in mega-pixel from string format.
"""
	
def front_camera_1(x):
	if (type(x)!=float and len(ast.literal_eval(x))==1):
		x=float(ast.literal_eval(x)[0].split()[0])
	else:
		x
	return x

"""
Extracting value of 2nd front camera in mega-pixel from string format.
"""

def front_camera_2(x):
	try:
		if (type(x)!=float and len(ast.literal_eval(x))==2):
			x=float(ast.literal_eval(x)[-1].split()[0])
		elif (type(x)!=float and len(ast.literal_eval(x))==1):
			x=0
		else:
			x
	except:
		pass
	return x
"""
Extracting different feature from string format we cannot use techniques as one hot encoder as there is no labeling done yet all data is presnt in some string(mesh-textual) formats.
"""
def networking_feature_creation(x,i):
	x_list=x.split(',')

	if 'Dual Sim' in x_list:
		data['Dual Sim'][i]=1

	if ' 3G' in x_list:
		data['3G'][i]=1

	if ' 4G' in x_list:
		data['4G'][i]=1

	if ' VoLTE' in x_list:
		data['VoLTE'][i]=1

	if ' Wi-Fi' in x_list:
		data['Wi-Fi'][i]=1

	if ' IR Blaster' in x_list:
		data['IR Blaster'][i]=1

	if 'Single Sim' in x_list:
		data['Single Sim'][i]=1

	if ' 5G' in x_list:
		data['5G'][i]=1

	if ' NFC' in x_list:
		data['NFC'][i]=1

	if ' HDMI' in x_list:
		data['HDMI'][i]=1

	if 'Wi-Fi' in x_list:
		data['Wi-Fi'][i]=1

	if 'Quad Sim' in x_list:
		data['Quad Sim'][i]=1

	if 'Triple Sim' in x_list:
		data['Triple Sim'][i]=1

"""
Extracting rear cam values in pixel from a data string , as it conatins all values within same string.
"""
def rear_cam(x,i):
	try:
		j=1
		if (len(ast.literal_eval(x))==1):
			if 'Quad' in x.split():
				data['Quad_cam'][i]=ast.literal_eval(x)[0].split()[0]
				j=0
		  
			if 'Penta' in x.split():
				data['Penta_cam'][i]=ast.literal_eval(x)[0].split()[0]
				j=0

			if 'Quad' not in x.split() and 'Rear' in x.split():
				data['Single_cam'][i]=ast.literal_eval(x)[0].split()[0]
				j=0

		if (len(ast.literal_eval(x))==3):
			if 'Triple' in x.split():
		 		data['Triple_cam'][i]=ast.literal_eval(x)[0].split()[0]
				j=0

		if (len(ast.literal_eval(x))==2):
			if 'Dual' in x.split():
				data['Dual_cam'][i]=ast.literal_eval(x)[0].split()[0]
				j=0

		if j==1:
			data['Single_cam'][i]=13

	except:
		pass  



"""
checking for float  
"""
def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False




def to_csv(df, path):
	df.loc[-1] = df.dtypes
	df.index = df.index + 1
	df.sort_index(inplace=True)
	df.to_csv(path, index=False)






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset',help="address of original dataset with format '.csv' ")
	parser.add_argument('--cleaned_dataset',help="address of final dataset with format '.csv' ")
	
	dataset=parser.dataset
	cleaned_dataset=parser.cleaned_dataset

	data_cleaning(dataset,cleaned_dataset)


