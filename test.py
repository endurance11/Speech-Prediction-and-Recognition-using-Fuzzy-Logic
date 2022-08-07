import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import accuracy_score
from skfuzzy.control.controlsystem import ControlSystemSimulation
import csv
import tkinter
from tkinter import *
from tkinter import Label
from tkinter import PhotoImage,Image
from PIL import ImageTk, Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import tensorflow as tf
import datetime
import os
	
def Fuzzy():

	
	csv_file = str("voice.csv")
	df = pd.read_csv(csv_file)

	"""##Checking Null values"""

	df.isnull().sum()

	"""##Dataset"""

	df

	"""###String to Int conversion"""

	df['gender_val'] = pd.factorize(df.label)[0]
	#0 for male
	#1 for female

	df

	max_value = df. max() 
	print(max_value)

	min_value = df. min()
	print(min_value)

	"""##Correlation Matrix (graph)"""

	corrmat=df.corr()
	cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1);
	plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)
	  
	cg

	corr = df.corr()['gender_val']
	corr.sort_values()

	"""## Filtering dataset"""

	new_df = df.filter(['Q25', 'meanfun','IQR','sp.ent','label','gender_val'])
	new_df
	#Q25 & meanfun have positive correlation
	#IQR and sp.ent have negative correlation

	"""##Scaling values"""

	scaler = MinMaxScaler()
	new_df[['Q25','meanfun','IQR','sp.ent']]=scaler.fit_transform(new_df[['Q25','meanfun','IQR','sp.ent']])

	new_df = new_df.round(3)
	new_df
	#rounding off min-max values to 3 decimal places

	new_df.describe()

	value_df = new_df.drop(columns= ['label','gender_val'])
	value_df
	#removing label coulumn

	x= value_df[['Q25','meanfun','IQR','sp.ent']].copy() 
	y= new_df[['label']].copy()

	"""##Decision Tree """

	classifier = DecisionTreeClassifier(criterion = 'entropy')
	classifier.fit

	"""###Train-Test Splitting"""

	#split the dataset-80% to training and 20% to testing
	x_train,x_test,y_train,y_test = train_test_split(value_df,new_df.gender_val,test_size=0.2,random_state=None)

	"""###Decision Tree Model"""

	#Instantitate the decision tree model
	dt = DecisionTreeClassifier(max_depth =4,criterion='entropy')

	#Build Model
	modl = dt.fit(x_train,y_train)

	predictedData = modl.predict(x_test)
	predictedData

	#for decision tree visualisation
	#str_new_df = new_df.applymap(str)
	#str_value_df = value_df.applymap(str)
	#feature = ['Q25','meanfun','IQR','sp.ent']

	dot_data=tree.export_graphviz(dt,out_file=None,feature_names=['Q25','meanfun','IQR','sp.ent'],class_names=['male','female'],filled=True,rounded=True,special_characters=True)

	"""###Decision Tree"""

	#Display the decision tree
	graph=graphviz.Source(dot_data)


	new_df.sort_values('label')

	"""##Fuzzy Logic"""

	Q25 = ctrl.Antecedent(np.arange(0,1,0.001),'Q25')
	meanfun = ctrl.Antecedent(np.arange(0,1,0.001),'meanfun')
	IQR = ctrl.Antecedent(np.arange(0,1,0.001),'IQR')
	sp_ent = ctrl.Antecedent(np.arange(0,1,0.001),'sp_ent')
	gender_val = ctrl.Consequent(np.arange(0,1.001,0.001),'gender_val')

	#Q25.automf(7) #as Q25 is required only 2 times in decision tree Q25 ≤ 0.104 and Q25 ≤ 0.855
	#meanfun.automf(7) # as meanfun is required 5 times in tree meanfun ≤ 0.479 meanfun ≤ 0.528 meanfun ≤ 0.678 meanfun ≤ 0.432 meanfun ≤ 0.406
	#IQR.automf(5)
	#sp_ent.automf(7)

	meanfun['low'] = fuzz.trimf(meanfun.universe,[0,0.211,0.422])
	meanfun['med'] = fuzz.trimf(meanfun.universe,[0.4,0.45,0.49])
	meanfun['high'] = fuzz.trimf(meanfun.universe,[0.476,0.485,0.53])
	meanfun['vhigh'] = fuzz.trimf(meanfun.universe,[0.5,0.75,1])

	IQR['low'] = fuzz.trimf(IQR.universe,[0.239,0.6,1])#male
	IQR['high'] = fuzz.trimf(IQR.universe,[0,0.15,0.31])#female

	sp_ent['low'] = fuzz.trimf(sp_ent.universe,[0,0.32,0.65])#male
	sp_ent['high'] = fuzz.trimf(sp_ent.universe,[0.45,0.7,1])#female


	Q25['low']=fuzz.trimf(Q25.universe,[0,0.31,0.634])#male
	Q25['high']=fuzz.trimf(Q25.universe,[0.2,0.6,1])#female

	gender_val['male']= fuzz.trimf(gender_val.universe,[-0.5,0,0.5])
	gender_val['female']=fuzz.trimf(gender_val.universe,[0.5,1,1.5])

	#gender_val['male'].view()
	#meanfun.view()
	#IQR.view()
	#sp_ent.view()
	#Q25.view()
	#dismal = 0.143
	#poor = 0.286
	#mediocre = 0.429
	#average = 0.572
	#decent = 0.715
	#good = 0.858
	#excellent = 1

	"""###Fuzzy rules"""

	rule1 = ctrl.Rule((meanfun['low'])&(IQR['high'])&(sp_ent['low']),gender_val['male'])
	#rule1.view()
	rule2= ctrl.Rule((meanfun['low'])&(IQR['high'])&(sp_ent['high']),gender_val['male'])

	rule3= ctrl.Rule((meanfun['low'])&(IQR['low']),gender_val['male'])

	rule4= ctrl.Rule((meanfun['med']|meanfun['high']|meanfun['vhigh'])&(Q25['low'])&(IQR['high']),gender_val['female'])

	rule4= ctrl.Rule((meanfun['med']|meanfun['high']|meanfun['vhigh'])&(Q25['low'])&(IQR['low']),gender_val['male'])

	rule5 =ctrl.Rule((meanfun['med']|meanfun['high']|meanfun['vhigh'])&(Q25['high'])&(sp_ent['low']),gender_val['female'])

	rule6 =ctrl.Rule((meanfun['med']|meanfun['high']|meanfun['vhigh'])&(Q25['high'])&(sp_ent['high'])&(IQR['high']),gender_val['female'])

	rule7 =ctrl.Rule((meanfun['med']|meanfun['high']|meanfun['vhigh'])&(Q25['high'])&(sp_ent['high'])&(IQR['low']),gender_val['male'])

	labelling_control=ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7])
	labelling = ctrl.ControlSystemSimulation(labelling_control)

	male=0
	female=0
	y=0
	spent=[]
	iqr=[]
	meanfun=[]
	y_true=[]
	q25=[]
	y_pred=[]
	gender_true=[]
	filename=open('input.csv')
	dff=csv.DictReader(filename)
	for col in dff:
		spent.append(col['spent'])
		meanfun.append(col['meanfun'])
		iqr.append(col['IQR'])
		q25.append(col['Q25'])
		gender_true.append(col['label'])
		
	for i in range(0,len(spent)):
		spent[i]=float(spent[i])
		meanfun[i]=float(meanfun[i])
		iqr[i]=float(iqr[i])
		q25[i]=float(q25[i])
	
	'''change range of samples from dataset'''	
	for i in range(100,1100):
		y=y+1
		y_true.append(gender_true[i])
		labelling.input['sp_ent'] = spent[i]
		labelling.input['IQR']  = iqr[i]
		labelling.input['meanfun']  = meanfun[i]
		labelling.input['Q25'] = q25[i]
		labelling.compute()
		if(labelling.output['gender_val']<0.5):
			male=male+1
			y_pred.append('male')
		elif(1>labelling.output['gender_val']>0.5):
			female=female+1
			y_pred.append('female')
		else:
			break

	accuracy=accuracy_score(y_true,y_pred)*100
	print("Number of Males is "+ str(male))
	print("Number of Females is " + str(female))
	print("Accuracy of Fuzzy approach is "+ str(accuracy))


	def graph():

		tree=Image.open("tree1.png")
		resize1=tree.resize((1600,470),Image.ANTIALIAS)
		img1 = ImageTk.PhotoImage(resize1)
		Label(window,image=img1).place(relx=0.08,rely=0.55)
		
		matrix=Image.open("matrix.png")
		resize2=matrix.resize((600,560),Image.ANTIALIAS)
		img2 = ImageTk.PhotoImage(resize2)
		Label(window,image=img2).place(relx=0.08,rely=0.01)
		
		one=Image.open("1.png")
		resize3=one.resize((500,275),Image.ANTIALIAS)
		img3=ImageTk.PhotoImage(resize3)
		Label(window,image=img3).place(relx=0.65,rely=0.01)
	
		mean=Image.open("meanfun.png")
		resize4=mean.resize((500,275),Image.ANTIALIAS)
		img4=ImageTk.PhotoImage(resize4)
		Label(window,image=img4).place(relx=0.65,rely=0.275)
			
		window.mainloop()
		
	
	global output	
		
	output = tkinter.Label(window, width='30', fg = "#fe2cbb",bg='#17005d',height='4',font=('Times New Roman',20))
	output.place(relx=0.42,rely=0.005)
	output.config(text="For " + str(y) + " samples \n number of males is " + str(male) + "\n and females is " + str(female) + "\n with " + str(accuracy) + "% accuracy")
	graph()
	window.mainloop()
	





def CNN():
	window.destroy()
	window.mainloop()



	log_folder='logs'

	csv_file = str("voice.csv")
	data = pd.read_csv(csv_file)

	data

	data.info()

	"""#Encoding Labels"""

	label_encoder = LabelEncoder()

	data['label'] = label_encoder.fit_transform(data['label'])

	dict(enumerate(label_encoder.classes_))

	data

	"""#Splitting and Scaling"""

	y = data['label'].copy()
	X = data.drop('label', axis=1).copy()

	scaler = StandardScaler()

	X = scaler.fit_transform(X)

	"""#CNN"""

	X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype=np.float, maxlen=25, padding='post')
	X = X.reshape(-1, 5, 5)
	X = np.expand_dims(X, axis=3)

	X.shape

	plt.figure(figsize=(12, 12))

	for i in range(9):
	    plt.subplot(3, 3, i + 1)
	    plt.imshow(np.squeeze(X[i]))
	    plt.axis('on')
	    
	manager = plt.get_current_fig_manager()
	manager.full_screen_toggle()    
	plt.show()
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

	inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2], X.shape[3]))

	x = tf.keras.layers.Conv2D(16, 2, activation='relu')(inputs)
	x = tf.keras.layers.MaxPooling2D()(x)

	x = tf.keras.layers.Conv2D(32, 1, activation='relu')(x)
	x = tf.keras.layers.MaxPooling2D()(x)

	x = tf.keras.layers.Flatten()(x)

	x = tf.keras.layers.Dense(64, activation='relu')(x)

	outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

	model = tf.keras.Model(inputs, outputs)

	model.summary()

	print((X.shape[1], X.shape[2], X.shape[3]))
	
	
	
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])

	history = model.fit(X_train,y_train,validation_split=0.2,batch_size=32,epochs=100,callbacks = [TensorBoard(log_dir=log_folder, histogram_freq=1,write_graph=True,write_images=True,update_freq='epoch',profile_batch=2)])

	"""#Prediction"""

	y_pre=model.predict(X_test[:20])
	new_y_pre = []
	for i in y_pre:
	  if i>0.5:
	    new_y_pre.append(1)
	  else:
	    new_y_pre.append(0)
	new_y_pre

	y_test[:20]

	tf.keras.utils.plot_model(model,"Layer.png",show_shapes=True)

	#Commented out IPython magic to ensure Python compatibility.
	#%tensorboard --logdir={log_folder}
	os.system('python -m tensorflow.tensorboard --logdir=' + "/home/yash/Downloads")
	
	
	
	
def accgraph():

	CNN=[95,96,97,98,98]
	Fuzzy=[89,91,95,96,97.5]
	x=[100,400,600,800,1000]
	# plot lines
	plt.plot(x,CNN,color="orange", label = "CNN", linestyle="-",linewidth=2.5)
	plt.plot(x,Fuzzy,color="blue", label = "Fuzzy", linestyle="--",linewidth=2.5)
	plt.title("Accuracy Comparison CNN vs Fuzzy")
	plt.ylabel("Percentage Accuracy")
	plt.xlabel("Sample Size")
	plt.legend()
	manager = plt.get_current_fig_manager()
	manager.full_screen_toggle()  
	plt.show()

	



"""GUI"""
window = tkinter.Tk()
window.title("SPEECH RECOGNITION AND PREDICTION")   
window.attributes('-fullscreen',True)
bg=Image.open("speech.png")
bgimg=ImageTk.PhotoImage(bg)
Label(window,image=bgimg).place(x=0,y=0,relwidth=1,relheight=1)

global output1
output1 = tkinter.Label(window, width='50', fg = "#fe2cbb",bg='#17005d',height='2',font=('Times New Roman',30))
output1.place(relx=0.235,rely=0.01)
output1.config(text="Which approach do you wish to go forward with?")

run=Button(window,text="Fuzzy Logic" ,bg="#cc5858" ,fg="black" ,width='15', height='2' ,font=("Times New Roman",18),command=lambda:[run.destroy(),output1.destroy(),run1.destroy(),run2.destroy(),window.update(),Fuzzy()],highlightbackground="black")
run.place(relx=0.345,rely=0.15,anchor='center')


run1=Button(window,text="CNN" ,bg="#cc5858" ,fg="black" ,width='15', height='2' ,font=("Times New Roman",18),command=lambda:[run1.destroy(),output1.destroy(),run.destroy(),run2.destroy(),window.update(),CNN()],highlightbackground="black")
run1.place(relx=0.645,rely=0.15,anchor='center')


run2=Button(window,text="View CNN vs Fuzzy accuracy graph?" ,bg="#cc5858" ,fg="black" ,width='30', height='2' ,font=("Times New Roman",18),command=lambda:[accgraph()],highlightbackground="black")
run2.place(relx=0.495,rely=0.92,anchor='center')



window.mainloop()

		
		
	
	

	

