

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

window = tkinter.Tk()
window.title("SPEECH RECOGNITION AND PREDICTION")   
#window.geometry('1280x720')
window.attributes('-fullscreen',True)

bg=Image.open("muj2.png")
bgimg=ImageTk.PhotoImage(bg)
Label(window,image=bgimg).place(x=0,y=0,relwidth=1,relheight=1)

"""##Dataset"""

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
#print(max_value)

min_value = df. min()
#print(min_value)

"""##Correlation Matrix (graph)"""

corrmat=df.corr()
cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1);
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)
#plt.show()
  
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
graph

new_df.sort_values('label')

"""##Fuzzy Logic"""

Q25 = ctrl.Antecedent(np.arange(0,1,0.001),'Q25')
meanfun = ctrl.Antecedent(np.arange(0,1,0.001),'meanfun')
IQR = ctrl.Antecedent(np.arange(0,1,0.001),'IQR')
sp_ent = ctrl.Antecedent(np.arange(0,1,0.001),'sp_ent')
gender_val = ctrl.Consequent(np.arange(0,1.001,0.001),'gender_val')

Q25.automf(7) #as Q25 is required only 2 times in decision tree Q25 ≤ 0.104 and Q25 ≤ 0.855
meanfun.automf(7) # as meanfun is required 5 times in tree meanfun ≤ 0.479 meanfun ≤ 0.528 meanfun ≤ 0.678 meanfun ≤ 0.432 meanfun ≤ 0.406
IQR.automf(7)
sp_ent.automf(7)

gender_val['male']= fuzz.trapmf(gender_val.universe,[-0.25,0,0.25,0.5])
gender_val['female']=fuzz.trapmf(gender_val.universe,[0.5,0.75,1,1.25])

IQR
#dismal = 0.143
#poor = 0.286
#mediocre = 0.429
#average = 0.572
#decent = 0.715
#good = 0.858
#excellent = 1

"""###Fuzzy rules"""

rule1 = ctrl.Rule((IQR['poor']|IQR['dismal']) & (meanfun['mediocre']|meanfun['poor'] |meanfun['dismal']) & (sp_ent['decent']|sp_ent['good']|sp_ent['excellent'])  , gender_val['female'])
#rule1.view()

rule2 = ctrl.Rule((IQR['poor']|IQR['dismal']) & (meanfun['mediocre']|meanfun['poor'] |meanfun['dismal']) &(sp_ent['average']|sp_ent['poor']|sp_ent['mediocre']|sp_ent['dismal']) , gender_val['male'])

rule3 = ctrl.Rule((IQR['poor']|IQR['dismal']) & (meanfun['mediocre']) &(sp_ent['average']|sp_ent['mediocre']) , gender_val['male'])

rule4 = ctrl.Rule((IQR['poor']) & (meanfun['average']|meanfun['decent']|meanfun['good']|meanfun['excellent']) &(sp_ent['average']|sp_ent['mediocre']) , gender_val['female'])

rule5 = ctrl.Rule((meanfun['mediocre']|meanfun['poor'] |meanfun['dismal']|meanfun['average'])&(sp_ent['average']|sp_ent['poor']|sp_ent['mediocre']|sp_ent['dismal']|sp_ent['decent']),gender_val['male'])

rule6 = ctrl.Rule((meanfun['mediocre']|meanfun['poor'] |meanfun['dismal']|meanfun['average'])&(sp_ent['good']|sp_ent['excellent'])&(IQR['average']|IQR['poor']|IQR['mediocre']|IQR['dismal']),gender_val['female'])


labelling_control=ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6])
labelling = ctrl.ControlSystemSimulation(labelling_control)

male=0
female=0
other=0
spent=[]
iqr=[]
meanfun=[]
y_true=[]
y_pred=[]
gender_true=[]
filename=open('input.csv')
dff=csv.DictReader(filename)
for col in dff:
	spent.append(col['spent'])
	meanfun.append(col['meanfun'])
	iqr.append(col['IQR'])
	gender_true.append(col['label'])
	
for i in range(0,len(spent)):
	spent[i]=float(spent[i])
	meanfun[i]=float(meanfun[i])
	iqr[i]=float(iqr[i])
	
for i in range(1000,1200):
	y_true.append(gender_true[i])
	labelling.input['sp_ent'] = spent[i]
	labelling.input['IQR']  = iqr[i]
	labelling.input['meanfun']  = meanfun[i]
	labelling.compute()
	if(labelling.output['gender_val']<0.5):
		male=male+1
		y_pred.append('male')
	elif(1>labelling.output['gender_val']>0.5):
		female=female+1
		y_pred.append('female')
	else:
		other =other+1

accuracy=accuracy_score(y_true,y_pred)*100
print("Male is "+ str(male))
print("Female is " + str(female))
print("Other is " + str(other))

def graph():

	tree=Image.open("tree.png")
	resize1=tree.resize((1600,470),Image.ANTIALIAS)
	img1 = ImageTk.PhotoImage(resize1)
	Label(window,image=img1).place(relx=0.08,rely=0.55)
	
	matrix=Image.open("matrix.png")
	resize2=matrix.resize((600,560),Image.ANTIALIAS)
	img2 = ImageTk.PhotoImage(resize2)
	Label(window,image=img2).place(relx=0.08,rely=0.01)
	
	trimf=Image.open("trimf.png")
	resize3=trimf.resize((500,275),Image.ANTIALIAS)
	img3=ImageTk.PhotoImage(resize3)
	Label(window,image=img3).place(relx=0.65,rely=0.01)
	
	output=Image.open("output.png")
	resize4=output.resize((500,275),Image.ANTIALIAS)
	img4=ImageTk.PhotoImage(resize4)
	Label(window,image=img4).place(relx=0.65,rely=0.275)
	
	window.mainloop()
	
def execute():	
	global output
	output.config(text="For 200 male samples \n number of males is\n " + str(male) + " and females is " + str(female) + " with\n " + str(accuracy) + "% accuracy")

run=Button(window,text="E\nx\ne\nc\nu\nt\ne" ,bg="#cc5858" ,fg="black" ,width='2', height='8' ,font=("Times New Roman",18), command = lambda:[execute(),graph()],highlightbackground= "black")
run.place(relx=0.025,rely=0.5,anchor='center')

output = tkinter.Label(window, width='25', fg = "black",bg='#59a9c8',height='4',font=('Times New Roman',18))
output.place(relx=0.42,rely=0.005)

window.mainloop()

