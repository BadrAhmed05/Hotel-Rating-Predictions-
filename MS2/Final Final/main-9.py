#code written by TEAM SC
import pickle
import tkinter as tk
import pandas as pd
from tkinter import filedialog, ttk
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model  import LinearRegression, LogisticRegression , Ridge ,ElasticNet ,Lasso
from sklearn.preprocessing import LabelEncoder ,PolynomialFeatures ,MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import re
from sklearn.metrics import accuracy_score

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
from joblib import dump, load
from tkinter import *
from tkinter import ttk
import time
import datetime
from PIL import ImageTk,Image
import os
import sqlite3
from tkinter import messagebox
import pandas as pd

import xgboost as xgb
from joblib import dump, load
train_data = pd.read_csv("D:\\Collage\\Sem 6\\machine learning\\Project\\MS2\\hotel-classification-dataset.csv")
def handle_duplicated(data):
    print(data.duplicated().sum())
    data = data.drop_duplicates()
    print(len(data))
    return data


train_data = handle_duplicated(train_data)
for i, col in enumerate(train_data.select_dtypes(include=['number']).columns):
    # axs[i, 0].scatter(data.index, data[col])
    # axs[i, 0].set_xlabel('Index')
    # axs[i, 0].set_ylabel(col)

    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train_data[(train_data[col] < lower_bound) | (train_data[col] > upper_bound)]
    train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]



X = train_data.iloc[:, :-1]
Y = train_data['Reviewer_Score']

#Data Spliting 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def handle_nulls_in_train_data(train_data):
    fill_values = {}
    for column in train_data.columns:
        if train_data[column].dtype == 'object':
            fill_value = train_data[column].mode()[0]
            fill_values[column] = fill_value
            train_data[column].fillna(fill_value, inplace=True)
        else:
            fill_value = train_data[column].median()
            fill_values[column] = fill_value
            train_data[column].fillna(fill_value, inplace=True)
    return train_data, fill_values

def handle_nulls_in_test_data(test_data, fill_values):
    for column in test_data.columns:
        if test_data[column].dtype == 'object':
            test_data[column].fillna(fill_values[column], inplace=True)
        else:
            test_data[column].fillna(fill_values[column], inplace=True)
    return test_data

# Load the training dataset
# train_data = pd.read_csv('train.csv')


# # Preprocessing Functions 

# ## Get Date 

def change_date(data, col):
    data[col] = pd.to_datetime(data[col])
    data['year'] = data[col].dt.year.astype(float)
    data['month'] = data[col].dt.month.astype(float)
    data['day'] = data[col].dt.day.astype(float)
    data = data.drop(['Review_Date'],axis= 1 )
    return data 


# 
# ## Get Address


def get_address(data):
    col =data['Hotel_Address']
    x=data['Hotel_Address'].str[-14:]
    data['Hotel_Address'] = data['Hotel_Address'].apply(lambda x: x.split()[-1])
    data['Hotel_Address'] =data['Hotel_Address'].apply(lambda x: 'United Kingdom' if x == 'Kingdom' else x)
    data['Hotel_Address'] =data['Hotel_Address'].apply(lambda x: 'United States' if x == 'States' else x)
    return data 


# ## Get Days 

def get_daysNo(data):
    pattern = r'\d+'
    data['days_since_review'] = data['days_since_review'].apply(lambda x: int(re.findall(pattern, x)[0]))
    return data


# ## Get Tags



def get_tags(data):
  data['Tags'] = [ast.literal_eval(row) for row in data['Tags']]

  for index, row in data['Tags'].items():
    for name in row:
      if "trip" in name:
        data.at[index, 'Trip'] = name
      if ("room" in name.lower()) or ("suite" in name.lower()) or ("guestroom" in name.lower()) or ("studio" in name.lower()) or ("king" in name.lower()):
        data.at[index, 'Room'] = name
      if "night" in name:
        data.at[index, 'Nights'] = name
  data = data.drop('Tags', axis=1)
  return data


def Feature_Encoder(data, label, encoders=None):
    if encoders is None:
        encoders = {}
    
    cols = ('Hotel_Name', 'Reviewer_Nationality', 'Room', 'Trip', 'Nights', 'Hotel_Address', 'Positive_Review', 'Negative_Review')

    if label:
        for column in cols:
            if column in data.columns:
                encoders[column] = LabelEncoder()
                data[column] = encoders[column].fit_transform(data[column])
                print(f"Encoded column: {column}")
                print(encoders)
            else:
                print(f"Column '{column}' not found in the dataset.")
    else:
        for col in cols:
            if col in data.columns:
                if col in encoders:
                    new_values = data[col].unique()
                    unseen_values = np.setdiff1d(new_values.astype(str), encoders[col].classes_.astype(str))
                    if len(unseen_values) > 0:
                        encoders[col].classes_ = np.append(encoders[col].classes_, unseen_values)
                    data[col] = encoders[col].transform(data[col])
                else:
                    print(f"LabelEncoder not found for column '{col}'.")
            else:
                print(f"Column '{col}' not found in the test dataset.")

    return data, encoders


# # Scaling Data


def scaler_fit_transform(X_train):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    # Save scaler object to disk
    dump(scaler, 'scaler.joblib')
    
    return X_train_scaled

def scaler_transform(X_test):
    # Load scaler object from disk
    scaler = load('scaler.joblib')
    
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_test_scaled

def pre(data,fillvalues,encoder,label,test):
					if label:
						data = change_date(data,'Review_Date')
						print('done date')
						print('done nulls')
						data = get_address(data)
						print('done add')
						data = get_daysNo(data)
						print('done day')
						data = get_tags(data)
						print('done tags')
						data ,fill_values= handle_nulls_in_train_data(data)
						print('handel nulls done')
						print(data.info())
						data , encoders = Feature_Encoder(data, label=True)
						print('encoding done in train')
						print(data.info())
						data = scaler_fit_transform(data)

						return data ,fill_values,encoders

					else:
						if (test):  
							data['Reviewer_Score'] = data['Reviewer_Score'].map({'Low_Reviewer_Score': 0,'Intermediate_Reviewer_Score': 1, 'High_Reviewer_Score': 2})
							y_test = data['Reviewer_Score'] 
							data = data.iloc[:, :-1]
							data = change_date(data,'Review_Date')
							print('done date')
							print('done nulls')
							data = get_address(data)
							print('done add')
							data = get_daysNo(data)
							print('done day')
							data = get_tags(data)
							print('done tags')
							data = handle_nulls_in_test_data(data, fillvalues)
							print(data.info())
							data ,_ = Feature_Encoder(data,label=False, encoders=encoder)
							print('done encoder in test')
							print(data.info())
							return data ,y_test
						else:
							data = change_date(data,'Review_Date')
							print('done date')
							print('done nulls')
							data = get_address(data)
							print('done add')
							data = get_daysNo(data)
							print('done day')
							data = get_tags(data)
							print('done tags')
							data = handle_nulls_in_test_data(data, fillvalues)
							print(data.info())
							data ,_ = Feature_Encoder(data,label=False, encoders=encoder)
							print('done encoder in test')
							print(data.info())
							data = scaler_transform(data)
							return data

X_train,fillvalues ,encoders = pre(X_train,0,0,1,0)
X_test = pre(X_test,fillvalues,encoders,0,0)




def Logistic_Regression_model(X_train,Y_train,X_test,Y_test):
    start = time.time()
    model = LogisticRegression(C=1, max_iter=1000).fit(X_train, Y_train)
    with open('logistic_regression_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    end = time.time()
    training_time = end - start
    start1 = time.time()
    with open('logistic_regression_model.pkl', 'rb') as file:
        lr_loaded = pickle.load(file)
    y_pred = lr_loaded.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    end1 = time.time()
    test_time = end1 - start1
    # print("Accuracy = ", accuracy * 100, '%')
    # print('Training Time = ', training_time, 's')
    # print('Test Time = ', test_time, 's')

    return accuracy, training_time, test_time


def Decision_Tree_model(X_train, Y_train, X_test, Y_test):
    start_time = time.time()
    rf_classifier = nltk.DecisionTreeClassifier(max_depth=10)
    Y_train = np.ravel(Y_train)
    rf_classifier.fit(X_train, Y_train)
    with open('decision_tree_model.pkl', 'wb') as file:
        pickle.dump(rf_classifier, file)
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    with open('decision_tree_model.pkl', 'rb') as file:
        dt_loaded = pickle.load(file)
    y_pred = dt_loaded.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    accuracy = accuracy_score(Y_test, y_pred)

    # print("Accuracy = ", accuracy * 100, '%')
    # print('Training Time = ', training_time, 's')
    # print('Test Time = ', test_time, 's')

    return accuracy, training_time, test_time

def Random_Forest_model(X_train,Y_train,X_test,Y_test):
    start_time = time.time()
    rf_classifier = RandomForestClassifier(max_depth=10)
    Y_train = np.ravel(Y_train)
    rf_classifier.fit(X_train, Y_train)
    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(rf_classifier, file)
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    with open('random_forest_model.pkl', 'rb') as file:
        rf_loaded = pickle.load(file)
    y_pred = rf_loaded.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    accuracy = accuracy_score(Y_test, y_pred)
    # filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    # print("Accuracy = ", accuracy*100,'%')
    # print('Training Time = ',training_time,'s')
    # print('Test Time = ', test_time,'s')

    return accuracy, training_time, test_time

def Gradient_Boosting_model(X_train,Y_train,X_test,Y_test):
    start_time = time.time()
    # Create an instance of the GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier()

    # Fit the model to the training data
    Y_train = np.ravel(Y_train)
    gb_classifier.fit(X_train, Y_train)
    with open('Gradient_Boosting_model.pkl', 'wb') as file:
        pickle.dump(gb_classifier, file)
    end_time = time.time()
    training_time = end_time - start_time

    # Predict on the test data
    start_time = time.time()
    with open('Gradient_Boosting_model.pkl', 'rb') as file:
        gb_loaded = pickle.load(file)
    y_pred = gb_loaded.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time

    # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    # print("Accuracy = ", accuracy * 100,'%')
    # print('Training Time = ', training_time, 's')
    # print('Test Time = ', test_time, 's')
    return accuracy, training_time, test_time

def xgb_model(X_train,Y_train,X_test,Y_test):
    start_time = time.time()
    # Create an instance of the GradientBoostingClassifier
    gb_classifier = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,


)

    # Fit the model to the training data
    Y_train = np.ravel(Y_train)
    gb_classifier.fit(X_train, Y_train)
    with open('xgb_model.pkl', 'wb') as file:
        pickle.dump(gb_classifier, file)
    end_time = time.time()
    training_time = end_time - start_time

    # Predict on the test data
    start_time = time.time()
    with open('xgb_model.pkl', 'rb') as file:
        gb_loaded = pickle.load(file)
    y_pred = gb_loaded.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time

    # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    # print("Accuracy = ", accuracy * 100,'%')
    # print('Training Time = ', training_time, 's')
    # print('Test Time = ', test_time, 's')
    return accuracy, training_time, test_time


# # # calling models 


accuracy_rf, training_time_rf, test_time_rf = Random_Forest_model(X_train,y_train,X_test,y_test)
accuracy_gb, training_time_gb, test_time_gb = Gradient_Boosting_model(X_train,y_train,X_test,y_test)
accuracy_lr, training_time_lr, test_time_lr = Logistic_Regression_model(X_train,y_train,X_test,y_test)
accuracy_dt, training_time_dt, test_time_dt = Decision_Tree_model(X_train,y_train,X_test,y_test)
accuracy_xg, training_time_xg, test_time_xg = xgb_model(X_train,y_train,X_test,y_test)


# # # Ploting 

# # 
print("=====================================")
print("Logistic Regression Accuracy = {:.2%}".format(accuracy_lr))
print("=====================================")
# print("Gradient Boosting Accuracy = {:.2%}".format(accuracy_gb))
# print("=====================================")
print("Decision Tree Accuracy = {:.2%}".format(accuracy_dt))
print("=====================================")
print("XGB Accuracy = {:.2%}".format(accuracy_xg))
print("=====================================")
print("Random Forest Accuracy = {:.2%}".format(accuracy_rf))
print("=====================================")


# # Print the Training time for each model
# print("=====================================")
print(f"Logistic Regression Training Time:  (Time: {training_time_lr:.2f} seconds)")
print("=====================================")
# print(f"Gradient Boosting Training Time:  (Time: {training_time_gb:.2f} seconds)")
# print("=====================================")
print(f"Decision Tree Training Time:  (Time: {training_time_dt:.2f} seconds)")
print("=====================================")
print(f"XGB Accuracy Training Time:  (Time: {training_time_xg:.2f} seconds)")
print("=====================================")
print(f"Random Forest Training Time:  (Time: {training_time_rf:.2f} seconds)")
print("=====================================")
# # Print the Test time for each model
# print("=====================================")
print(f"Logistic Regression Testing Time:  (Time: {test_time_lr:.2f} seconds)")
print("=====================================")
# print(f"Gradient Boosting Testing Time:  (Time: {test_time_gb:.2f} seconds)")
# print("=====================================")
print(f"Decision Tree Testing Time:  (Time: {test_time_dt:.2f} seconds)")
print("=====================================")
print(f"XGB Accuracy Testing Time:  (Time: {test_time_xg:.2f} seconds)")
print("=====================================")
print(f"Random Forest Testing Time:  (Time: {test_time_rf:.2f} seconds)")
print("=====================================")
# ##################
models = ['Logistic Regression', 'Gradient Boosting', 'Decision Tree', 'XGB Accuracy','Random Forest']
mse_values = [accuracy_lr,accuracy_gb,accuracy_dt,accuracy_xg,accuracy_rf]

test , y_t = pre(test,fillvalues,encoders,0,1)


def Loadmodel(X_test,Y_test):
    # start_time = time.time()
    # with open('svm_model.pkl', 'rb') as file:
    #     gb_loaded = pickle.load(file)
    # y_pred = gb_loaded.predict(X_test)
    # end_time = time.time()
    # test_time = end_time - start_time

    # # Evaluate the model
    # accuracy = accuracy_score(Y_test, y_pred)
    
    start_time = time.time()
    with open('xgb_model.pkl', 'rb') as file:
        gb_loaded = pickle.load(file)
    y_pred = gb_loaded.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time

    # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy = ", accuracy * 100,'%')
    # print('Training Time = ', training_time, 's')
    # print('Test Time = ', test_time, 's')

    start_time = time.time()
    with open('Gradient_Boosting_model.pkl', 'rb') as file:
        gb_loaded = pickle.load(file)
    y_pred = gb_loaded.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    
     # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy = ", accuracy * 100,'%')
     # print('Training Time = ', training_time, 's')
     # print('Test Time = ', test_time, 's')
 
    start_time = time.time()
    with open('random_forest_model.pkl', 'rb') as file:
        rf_loaded = pickle.load(file)
    y_pred = rf_loaded.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    accuracy = accuracy_score(Y_test, y_pred)
    # filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    print("Accuracy = ", accuracy*100,'%')
    # print('Training Time = ',training_time,'s')
    # print('Test Time = ', test_time,'s')


    start1 = time.time()
    with open('logistic_regression_model.pkl', 'rb') as file:
        lr_loaded = pickle.load(file)
    y_pred = lr_loaded.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    end1 = time.time()
    test_time = end1 - start1
    print("Accuracy = ", accuracy * 100, '%')
    # print('Training Time = ', training_time, 's')
    # print('Test Time = ', test_time, 's')

		
    start_time = time.time()
    with open('decision_tree_model.pkl', 'rb') as file:
        loaded = pickle.load(file) 
        y_pred = dt_loaded.predict(X_test) 
        end_time = time.time() 
        test_time = end_time - start_time
        accuracy = accuracy_score(Y_test, y_pred) 
        messagebox.showinfo("Acuuracy ",accuracy) 
        messagebox.showinfo("Time for test ",test_time)




# Set the width of the bars
bar_width = 0.35

# Create a numpy array for the x-axis positions of the bars
x_pos = range(len(models))

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the bars
bars = ax.bar(x_pos, mse_values, bar_width)

# Add labels, title, and legend
ax.set_xlabel('Classification Models')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Classification Models')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

# Show the plot
plt.show()

n_groups = 5
means_frank = (training_time_lr, training_time_gb, training_time_dt, training_time_xg, training_time_rf)
means_guido = (test_time_lr, test_time_gb, test_time_dt, test_time_xg, test_time_rf)


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2  # Reduce the bar width
opacity = 0.8

rects1 = ax.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Training Time')

rects2 = ax.bar(index + bar_width, means_guido, bar_width,alpha=opacity,color='g',label='Testing Time')

ax.set_xlabel('Model')
ax.set_ylabel('Time (s)')
ax.set_title('Training and Testing Time by Model')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('LogisticR', 'Gradient', 'Decision', 'XGB', 'Random'))
ax.legend()

fig.tight_layout()
plt.show()


arr=get_address(train_data)
hotel_names = arr.groupby('Hotel_Address')['Hotel_Name'].unique()
# now = datetime.datetime.now()
#----------- importing sqlite for server side operations---------------------------------------------------------------------------------
con = sqlite3.Connection('hm_proj.db')
cur = con.cursor()
cur.execute("create table if not exists hoteld(t_r number,r_r number,t_s number)")
cur.execute("create table if not exists roomd(rn number primary key,beds number,ac varchar(10),tv varchar(10),internet varchar(10),price number(10))")
cur.execute("create table if not exists payments(id number primary key,dot varchar(15),time varchar(10),amt number,method varchar(10))")
cur.execute("create table if not exists paymentsf(id number  primary key,f_name varchar,l_name varchar,c_number varchar,email varchar , r_n number ,day varchar,month varchar,year varchar,time varchar , method varchar,totalamt varchar)")
con.commit()
con.commit()
cur.execute("select * from payments")
con.commit()
x=cur.fetchall()
con.commit()
#-----------splash_screen------------------------------------------------------------------------------------------------------------------
sroot = Tk()
sroot.minsize(height=516,width=1150)
sroot.configure(bg='white')
spath = "machine power.png"
simg = ImageTk.PhotoImage(Image.open(spath))
my = Label(sroot,image=simg)
my.image = simg
my.place(x=150,y=0)
#----------- main project------------------------------------------------------------------------------------------------------------------
def mainroot():
	#sroot.destroy()
	root = Tk()
	root.geometry('1080x500')
	root.minsize(width=1080,height=550)
	root.maxsize(width=1080,height=550)
	root.configure(bg='white')
	root.title("Machine learning")
	#--------------seperator-------------------------------------------------------------------------------------------------------------------
	sep = Frame(height=500,bd=1,relief='sunken',bg='white')
	#----------------Connection with printer-------------------------------------------------------------------------------------------------------------
	def connectprinter():
		os.startfile("C:/Users/TestFile.txt", "print")
	#---------------top frame------------------------------------------------------------------------------------------------------------------
	top_frame = Frame(root,height=70,width=1080,bg='orange')
	path = "images/newestbg5.jpg"
	img = ImageTk.PhotoImage(Image.open(path))
	label = Label(top_frame,image = img ,height=70,width=1080)
	label.image=img
	label.place(x=0,y=0)
	top_frame.place(x=0,y=0)
	tf_label = Label(top_frame,text='Hotel Rating Predection ',font='msserif 33',fg='black',bg='gray89',height=70)
	tf_label.pack(anchor='center')
	top_frame.pack_propagate(False)
	#---------------DATE TIME------------------------------------------------------------------------------------------------------------------
	def datetime():
		#while(True):
		localtime = now.strftime("%Y-%m-%d %H:%M")
		lblInfo = Label(top_frame,font='helvetica 15',text=localtime,bg='blue',fg='white')
	#----------------bottom frame - hotel status and default page-------------------------------------------------------------------------------
	def hotel_status():
		b_frame = Frame(root, height=400, width=1080, bg='gray89')
		path = "images/newbg6lf.jpg"
		img = ImageTk.PhotoImage(Image.open(path))
		label = Label(b_frame, image=img, height=400, width=1080)
		label.image = img
		label.place(x=0, y=0)
		l = Label(b_frame, text='Please Enter The hotel name', font='msserif 15', bg='cyan4', fg='white')
		l.place(x=245, y=0)
		b_frame.place(x=0, y=120 + 6 + 20 + 60 + 11)
		b_frame.pack_propagate(False)
		b_frame.tkraise()
		hline = Frame(b_frame, height=42, width=1080, bg='cyan4')
		hline.place(x=0, y=23)
		ef = Frame(hline)
		p_id = Entry(ef)
		p_id.pack(ipadx=25, ipady=3)
		ef.place(x=308, y=6)
		fl1 = Frame(b_frame, height=38, width=308, bg='cyan4')
		fl1.place(x=0, y=68)
		l1 = Label(fl1, text='High_Reviewer_Score', bg='cyan4', fg='white', font='msserif 17')
		l1.pack()
		fl1.pack_propagate(False)
		fr1 = Frame(b_frame, height=38, width=1080 - 308, bg='white')
		fr1.place(x=0 + 308, y=68)
		l1 = Label(fl1, text='Date of transaction', bg='cyan4', fg='white', font='msserif 17')
		fr1.pack_propagate(False)
		fl2 = Frame(b_frame, height=38, width=308, bg='cyan4')
		fl2.place(x=0, y=109)
		fl2.pack_propagate(False)
		l1 = Label(fl2, text='low_Reviewer_Score', bg='cyan4', fg='white', font='msserif 17')
		l1.pack()
		fr2 = Frame(b_frame, height=38, width=1080 - 308, bg='white')
		fr2.place(x=0 + 308, y=109)
		fr2.pack_propagate(False)
		l1 = Label(fl1, text='Date of transaction', bg='cyan4', fg='white', font='msserif 17')
		fl3 = Frame(b_frame, height=38, width=308, bg='cyan4')
		fl3.place(x=0, y=150)
		fl3.pack_propagate(False)
		l1 = Label(fl3, text='Intermediate_Reviewer_Score', bg='cyan4', fg='white', font='msserif 17')
		l1.pack()
		fr3 = Frame(b_frame, height=38, width=1080 - 308, bg='white')
		fr3.place(x=0 + 308, y=150)
		fr3.pack_propagate(False)
		l1 = Label(fl1, text='Date of transaction', bg='cyan4', fg='white', font='msserif 17')



		def getid(event=None):
			fl1 = Frame(b_frame, height=38, width=308, bg='cyan4')
			fl1.place(x=0, y=68)
			l1 = Label(fl1, text='High_Reviewer_Score', bg='cyan4', fg='white', font='msserif 17')
			l1.pack()
			fl1.pack_propagate(False)
			fr1 = Frame(b_frame, height=38, width=1080 - 308, bg='white')
			fr1.place(x=0 + 308, y=68)
			l1 = Label(fl1, text='Date of transaction', bg='cyan4', fg='white', font='msserif 17')
			fr1.pack_propagate(False)
			fl2 = Frame(b_frame, height=38, width=308, bg='cyan4')
			fl2.place(x=0, y=109)
			fl2.pack_propagate(False)
			l1 = Label(fl2, text='low_Reviewer_Score', bg='cyan4', fg='white', font='msserif 17')
			l1.pack()
			fr2 = Frame(b_frame, height=38, width=1080 - 308, bg='white')
			fr2.place(x=0 + 308, y=109)
			fr2.pack_propagate(False)
			l1 = Label(fl1, text='Date of transaction', bg='cyan4', fg='white', font='msserif 17')
			fl3 = Frame(b_frame, height=38, width=308, bg='cyan4')
			fl3.place(x=0, y=150)
			fl3.pack_propagate(False)
			l1 = Label(fl3, text='Intermediate_Reviewer_Score', bg='cyan4', fg='white', font='msserif 17')
			l1.pack()
			fr3 = Frame(b_frame, height=38, width=1080 - 308, bg='white')
			fr3.place(x=0 + 308, y=150)
			fr3.pack_propagate(False)
			idd = p_id.get()
			data = pd.read_csv("hotel-classification-dataset.csv")
			positive_count = 0
			negative_count = 0
			normal_count = 0
			count = 0
			# Group the data by unique Hotel_Name and get the unique Reviewer_Score in each group
			# hotel_names = data.groupby('Hotel_Name')['Reviewer_Score'].unique()
			for i in data["Hotel_Name"]:
				if (i == idd):
					for z in data["Reviewer_Score"]:
						if z == "High_Reviewer_Score":
							positive_count += 1
						elif z == "Low_Reviewer_Score":
							negative_count += 1
						elif z == "Intermediate_Reviewer_Score":
							normal_count += 1
			yy=[positive_count,negative_count,normal_count]
			print(yy)
			if (yy != None):
				dot = yy[0]
				tot = yy[1]
				ap = yy[2]
				pm = '--'
				l1 = Label(fr1, text=dot, height=38, width=1080 - 308, font='msserif 15', bg='white', fg='cyan4').pack()
				l2 = Label(fr2, text=tot, height=38, width=1080 - 308, font='msserif 15', bg='white', fg='cyan4').pack()
				l3 = Label(fr3, text=ap, height=38, width=1080 - 308, font='msserif 15', bg='white', fg='cyan4').pack()

			else:
				l1 = Label(fr1, text='No Information Available', height=38, width=1080 - 308, font='msserif 15',
						   bg='white', fg='cyan4').pack()
				l1 = Label(fr2, text='No Information Available', height=38, width=1080 - 308, font='msserif 15',
						   bg='white', fg='cyan4').pack()
				l1 = Label(fr3, text='No Information Available', height=38, width=1080 - 308, font='msserif 15',
						   bg='white', fg='cyan4').pack()


		ok = Button(hline, text='OK', font='msserif 10', bg='white', activebackground='steelblue', fg='cyan4',
					command=getid)
		ok.place(x=530, y=5)
		p_id.bind('<Return>', getid)

		def pr():
			messagebox.askyesno("Print", "Do you want to print Reciept")

		# pinv = Button(b_frame, text='Print', bg='Green', fg='white', command=pr).place(x=976, y=235)
		# ------------inner frames of bottom frame-------------------------
		nl = Label(b_frame, text='Made by Avengers', fg='black', bg='gray91', font='msserif 8')
		nl.place(x=955, y=310)
		nl.tkraise()
	#-------------- Guests --------------------------------------------------------------------------------------------------------------------------
	def staff():
		b_frame = Frame(root, height=400, width=1080, bg='white')

		emp1inf = Frame(b_frame, bg='White', height=122, width=240)
		Label(emp1inf, text="Member 1", bg='white', font='msserif 17 bold').place(x=60, y=0)
		Label(emp1inf, text="Zeinab Asaad", bg='white', fg="Grey", font='msserif 10').place(x=60, y=37)
		Label(emp1inf, text="Mail : zeinabasaad973@gmail.com", bg='white', fg="Grey", font='msserif 10').place(x=60,
																											   y=59)
		emp1inf.pack_propagate(False)
		emp1inf.place(x=1, y=5)
		emp1f = Frame(b_frame)

		emp1inf = Frame(b_frame, bg='White', height=122, width=240)
		Label(emp1inf, text="Member 2", bg='white', font='msserif 17 bold').place(x=60, y=0)  # pack(side='top')
		Label(emp1inf, text="Mohand Mahmoud", bg='white', fg="Grey", font='msserif 10').place(x=60, y=37)
		Label(emp1inf, text="Mail : mohandmahmoudmousa@gmail.com", bg='white', fg="Grey", font='msserif 10').place(x=60,
																												   y=59)
		emp1inf.pack_propagate(False)
		emp1inf.place(x=300, y=5)
		emp1f = Frame(b_frame)

		emp1inf = Frame(b_frame, bg='White', height=122, width=240)
		Label(emp1inf, text="Member 3", bg='white', font='msserif 17 bold').place(x=60, y=0)  # pack(side='top')
		Label(emp1inf, text="Mohamed gamal", bg='white', fg="Grey", font='msserif 10').place(x=60, y=37)
		Label(emp1inf, text="Mail : mohamed@gmail.com", bg='white', fg="Grey", font='msserif 10').place(x=60, y=59)
		emp1inf.pack_propagate(False)
		emp1inf.place(x=600, y=5)
		emp1f = Frame(b_frame)

		emp1inf = Frame(b_frame, bg='White', height=122, width=240)
		Label(emp1inf, text="Member 4", bg='white', font='msserif 17 bold').place(x=60, y=0)  # pack(side='top')
		Label(emp1inf, text="Ziad nagi", bg='white', fg="Grey", font='msserif 10').place(x=60, y=37)
		Label(emp1inf, text="Mail : ziad@gmail.com", bg='white', fg="Grey", font='msserif 10').place(x=60, y=59)
		emp1inf.pack_propagate(False)
		emp1inf.place(x=1, y=200)
		emp1f = Frame(b_frame)

		emp1inf = Frame(b_frame, bg='White', height=122, width=240)
		Label(emp1inf, text="Member 5", bg='white', font='msserif 17 bold').place(x=60, y=0)  # pack(side='top')
		Label(emp1inf, text="Badr Ahmed", bg='white', fg="Grey", font='msserif 10').place(x=60, y=37)
		Label(emp1inf, text="Mail : badr@gmail.com", bg='white', fg="Grey", font='msserif 10').place(x=60, y=59)
		emp1inf.pack_propagate(False)
		emp1inf.place(x=300, y=200)
		emp1f = Frame(b_frame)

		emp1inf = Frame(b_frame, bg='White', height=122, width=240)
		Label(emp1inf, text="Member 6", bg='white', font='msserif 17 bold').place(x=60, y=0)  # pack(side='top')
		Label(emp1inf, text="Roaa Alaa", bg='white', fg="Grey", font='msserif 10').place(x=60, y=37)
		Label(emp1inf, text="Mail : roaa@gmail.com", bg='white', fg="Grey", font='msserif 10').place(x=60, y=59)
		emp1inf.pack_propagate(False)
		emp1inf.place(x=600, y=200)
		emp1f = Frame(b_frame)

		Frame(b_frame, height=13, width=250, bg='white').place(x=410, y=2)
		Frame(b_frame, height=13, width=250, bg='white').place(x=410, y=153)
		b_frame.place(x=0, y=120 + 6 + 20 + 60 + 11)
		b_frame.pack_propagate(False)
		b_frame.tkraise()
		nl = Label(b_frame, text='Made by Avengers', fg='black', bg='gray91', font='msserif 8')
		nl.place(x=955, y=310)
		nl.tkraise()
	#-------------- rooms --------------------------------------------------------------------------------------------------------------------------
	def rooms():
		b_frame = Frame(root,height=400,width=1080,bg='gray91')
		b_frame.place(x=0,y=120+6+20+60+11)
		b_frame.pack_propagate(False)
		b_frame.tkraise()
		path = "images/newbg6lf.jpg"
		img = ImageTk.PhotoImage(Image.open(path))
		label = Label(b_frame,image = img ,height=400,width=1080)
		label.image=img
		label.place(x=0,y=0)
		sidebuttons = Text(b_frame,width=1,height=19)
		sc = Scrollbar(b_frame,command=sidebuttons.yview,width=10,bg='lightsteelblue3')
		sidebuttons.configure(yscrollcommand=sc.set)
		sc.pack(side='left',fill=Y)
		sidebuttons.place(x=10,y=0)
		def roomdet(rno):
			Label(b_frame,text='Country',font='msserif 15',fg='white',bg='cyan4',width=10).place(x=570,y=0)
			smf1 = Frame(b_frame,height=1000,width=1000,bg='white')
			hline = Frame(b_frame,height=10,width=960,bg='cyan4')
			hline.place(x=122,y=27)
			vline = Frame(b_frame,height=400,width=7,bg='lightsteelblue3')
			vline.place(x=122,y=0) 
			tr = Label(smf1,text='Hotels:',fg='white',bg='cyan4',width=100,height=2,font='msserif 15')
			tr.pack(side='top')
			smf1.pack_propagate(False)
			smf1.place(x=129+3,y=30)
			hotels = {
				"United Kingdom": ["H tel Bedford", "Best Western Plus de Neuville Arc de Triomphe",
								   "Lyric H tel Paris", "Novotel Paris Gare De Lyon",
								   "Renaissance Paris Arc de Triomphe Hotel", "Hotel La Villa Saint Germain Des Pr s"],
				"Austria": ["Imperial Riding School Renaissance Vienna Hotel",
							"Austria Trend Hotel Schloss Wilhelminenberg Wien", "NH Danube City",
							"Austria Trend Hotel Park Royal Palace Vienna", "Hotel Capricorno", "Hotel Stefanie"],
				"Netherlands": ["Millennium Hotel London Mayfair", "Amba Hotel Marble Arch",
								"Holiday Inn London Oxford Circus", "London City Suites", "Sofitel London St James",
								"COMO Metropolitan London"],
				"Spain": ["Catalonia Atenas", "Exe Laietana Palace", "Catalonia Barcelona Plaza", "H10 Casanova",
						  "Hotel Well and Come", "Eurostars Grand Marina Hotel GL"],
				"France": ["De L Europe Amsterdam", "The Toren", "Swiss tel Amsterdam",
						   "Ramada Apollo Amsterdam Centre", "Jaz Amsterdam", "The College Hotel"],
				"Italy": ["Hotel Da Vinci", "Glam Milano", "Petit Palais Hotel De Charme",
						  "NH Collection Milano President", "Hotel Bristol", "UNA Hotel Scandinavia"]
			}
			for i in hotels:
				if (i == rno):
					for j in hotels[i]:
						Label(smf1, text=str(j), fg='cyan4', bg='white', font='msserif 15').pack()

		roomdet("")
		sidebuttons.configure(state='disabled')
		b1  = Button(b_frame,font='mssherif 10', text="United Kingdom", bg='white',fg='cyan4',width=12,command=lambda:roomdet("United Kingdom"))
		b2  = Button(b_frame,font='mssherif 10', text="Italy", bg='white',fg='cyan4',width=12,command=lambda:roomdet("Italy"))
		b3  = Button(b_frame,font='mssherif 10', text="France", bg='white',fg='cyan4',width=12,command=lambda:roomdet("France"))
		b4  = Button(b_frame,font='mssherif 10', text="Spain", bg='white',fg='cyan4',width=12,command=lambda:roomdet("Spain"))
		b5  = Button(b_frame,font='mssherif 10', text="Netherlands", bg='white',fg='cyan4',width=12,command=lambda:roomdet("Netherlands"))
		b6  = Button(b_frame,font='mssherif 10', text="Austria", bg='white',fg='cyan4',width=12,command=lambda:roomdet("Austria"))
		sidebuttons.window_create("end",window=b1)
		sidebuttons.insert("end","\n")
		sidebuttons.window_create("end",window=b2)
		sidebuttons.insert("end","\n")
		sidebuttons.window_create("end",window=b3)
		sidebuttons.insert("end","\n")
		sidebuttons.window_create("end",window=b4)
		sidebuttons.insert("end","\n")
		sidebuttons.window_create("end",window=b5)
		sidebuttons.insert("end","\n")
		sidebuttons.window_create("end",window=b6)
		sidebuttons.insert("end","\n")
		nl = Label(b_frame,text='Made by Avengers',fg='black',bg='gray91',font='msserif 8')
		nl.place(x=1000,y=310)
		nl.tkraise()
	#--------------- payments-----------------------------------------------------------------------------------------------------------------------
	def payments():
		b_frame = Frame(root,height=400,width=1080,bg='gray89')
		path = "images/newbg6lf.jpg"
		img = ImageTk.PhotoImage(Image.open(path))
		label = Label(b_frame,image = img ,height=400,width=1080)
		label.image=img
		label.place(x=0,y=0)
		b_frame.place(x=0,y=120+6+20+60+11)
		b_frame.pack_propagate(False)
		b_frame.tkraise()
		hline = Frame(b_frame,height=42,width=1080,bg='cyan4')
		hline.place(x=0,y=23)
		ef = Frame(hline)
		p_id = Entry(ef)
		p_id.pack(ipadx=25,ipady=3)
		ef.place(x=308,y=6)
		fl1=Frame(b_frame,height=38,width=308,bg='cyan4')
		fl1.place(x=0,y=68)
		def upload_test_file():
			f_types = [('CSV files', "*.csv"), ('All', "*.*")]
			file = filedialog.askopenfilename(filetypes=f_types)
			global test
			test = pd.read_csv(file)  # create DataFrame
	
		browse_btn = tk.Button(b_frame, text='Browse File',
                       width=20, command=lambda: upload_test_file())
		browse_btn.place(x = 325, y=30)
		# testing_frame_label1 = tk.Label(b_frame, text='Read Test File & create DataFrame',

		

		pre_btn = tk.Button(b_frame, text='Preprocessing',
					width=20, command=lambda: pre(test,fillvalues,encoders,0,1))

		pre_btn.place(x = 325, y=70)
		pre_btn = tk.Button(b_frame, text='Models',
		               width=20, command=lambda: Loadmodels())
		pre_btn.place(x = 325, y=110,
		                        width=30, font="msserif 13")
				# testing_frame_label1.grid(row=0, column=1)


		
				
		
		messagebox.showinfo("Done preprocessing")

		# test = pre(test,fillvalues,encoders,0,1)
	#---------------reserve------------------------------------------------------------------------------------------------------------------------
	def reserve():
		b_frame = Frame(root,height=420,width=1080,bg='gray89')
		path = "images/newbg6lf.jpg"
		img = ImageTk.PhotoImage(Image.open(path))
		label = Label(b_frame,image = img ,height=420,width=1080)
		label.image=img
		label.place(x=0,y=0)
		vline = Frame(b_frame,height=400,width=7,bg='lightsteelblue3')
		vline.place(x=700,y=0)
		Label(b_frame,text='Personal Information',font='msserif 15',bg='gray93').place(x=225,y=0)
		fnf = Frame(b_frame,height=1,width=1)
		fn = Entry(fnf)
		mnf = Frame(b_frame,height=1,width=1)
		mn = Entry(mnf)
		lnf = Frame(b_frame,height=1,width=1)
		ln = Entry(lnf)
		
		fn.pack(ipady=4,ipadx=15)
		mn.pack(ipady=4,ipadx=15)
		ln.pack(ipady=4,ipadx=15)
		fnf.place(x=20,y=42)
		mnf.place(x=235,y=42)
		lnf.place(x=450,y=42)
		Label(b_frame,text='Contact Information',font='msserif 15',bg='gray93').place(x=225,y=90)
		cnf = Frame(b_frame,height=1,width=1)
		cn = Entry(cnf)
		emf = Frame(b_frame,height=1,width=1)
		em = Entry(emf)
		adf = Frame(b_frame,height=1,width=1)
		ad = Entry(adf)
		cn.insert(0, 'Contact Number *')
		em.insert(0, 'Email *')
		ad.insert(0, "Guest's Address *")
		def on_entry_click4(event):
			if cn.get() == 'Contact Number *' :
				cn.delete(0,END)
				cn.insert(0,'')
		def on_entry_click5(event):
			if em.get() == 'Email *' :
				em.delete(0,END)
				em.insert(0,'')
		def on_entry_click6(event):
			if ad.get() == "Guest's Address *" :
				ad.delete(0,END)
				ad.insert(0,'')
		def on_exit4(event):
			if cn.get()=='':
				cn.insert(0,'Contact Number *')
		def on_exit5(event):
			if em.get()=='':
				em.insert(0,'Email *')
		def on_exit6(event):
			if ad.get()=='':
				ad.insert(0,"Guest's Address *")
		cn.bind('<FocusIn>', on_entry_click4)
		em.bind('<FocusIn>', on_entry_click5)
		ad.bind('<FocusIn>', on_entry_click6)
		cn.bind('<FocusOut>',on_exit4)
		em.bind('<FocusOut>',on_exit5)
		ad.bind('<FocusOut>',on_exit6)
		cn.pack(ipady=4,ipadx=15)
		em.pack(ipady=4,ipadx=15)
		ad.pack(ipady=4,ipadx=15)
		cnf.place(x=20,y=130)
		emf.place(x=235,y=130)
		adf.place(x=450,y=130)
		Label(b_frame,text='Reservation Information',font='msserif 15',bg='gray93').place(x=210,y=175)
		nocf = Frame(b_frame,height=1,width=1)
		noc = Entry(nocf)
		noaf = Frame(b_frame,height=1,width=1)
		noa = Entry(noaf)
		nodf = Frame(b_frame,height=1,width=1)
		nod = Entry(nodf)
		noc.insert(0, 'Number of Children *')
		noa.insert(0, 'Number of Adults *')
		nod.insert(0, 'Number of Days of Stay *')
		def on_entry_click7(event):
			if noc.get() == 'Number of Children *' :
				noc.delete(0,END)
				noc.insert(0,'')
		def on_entry_click8(event):
			if noa.get() == 'Number of Adults *' :
				noa.delete(0,END)
				noa.insert(0,'')
		def on_entry_click9(event):
			if nod.get() == 'Number of Days of Stay *' :
				nod.delete(0,END)
				nod.insert(0,'')
		def on_exit7(event):
			if noc.get()=='':
				noc.insert(0,'Number of Children *')
		def on_exit8(event):
			if noa.get()=='':
				noa.insert(0,'Number of Adults *')
		def on_exit9(event):
			if nod.get()=='':
				nod.insert(0,'Number of Days of Stay *')
		noc.bind('<FocusIn>', on_entry_click7)
		noa.bind('<FocusIn>', on_entry_click8)
		nod.bind('<FocusIn>', on_entry_click9)
		noc.bind('<FocusOut>',on_exit7)
		noa.bind('<FocusOut>',on_exit8)
		nod.bind('<FocusOut>',on_exit9)
		noc.pack(ipady=4,ipadx=15)
		noa.pack(ipady=4,ipadx=15)
		nod.pack(ipady=4,ipadx=15)
		nocf.place(x=20,y=220)
		noaf.place(x=235,y=220)
		nodf.place(x=450,y=220)
		roomnf = Frame(b_frame,height=1,width=1)
		roomn = Entry(roomnf)
		roomn.insert(0, 'Enter Room Number *')
		def on_entry_click10(event):
			if roomn.get() == 'Enter Room Number *' :
				roomn.delete(0,END)
				roomn.insert(0,'')
		def on_exit10(event):
			if roomn.get()=='':
				roomn.insert(0,'Enter Room Number *')	
		roomn.bind('<FocusIn>', on_entry_click10)
		roomn.bind('<FocusOut>',on_exit10)
		roomn.pack(ipady=4,ipadx=15)
		roomnf.place(x=20,y=270)
		pmethod = IntVar()
		def booking():
			if fn.get() == 'First Name' or ln.get() == 'Last Name' or cn.get() == 'Contact Number *' or em.get() == 'Email' or ad.get() == "Guest's Address" or noc.get() == 'Number of Children' or noa.get() == 'Number of Adults' or nod.get() == 'Number of Days of Stay' or roomn.get() == 'Enter Room Number':
				messagebox.showinfo('Incomplete','Fill All the Fields marked by *')
			elif fn.get() == '' or ln.get() == '' or cn.get() == '' or em.get() == '' or ad.get() == "" or noc.get() == '' or noa.get() == '' or nod.get() == '' or roomn.get() == '':
				messagebox.showinfo('Incomplete','Fill All the Fields marked by *')
			else :
				cur.execute("select rstatus from roomd where rn = ?",(roomn.get(),))
				temp = cur.fetchone()
				if temp[0] == 'Reserved':
					messagebox.showwarning('Room is Reserved','Room number '+str(roomn.get())+' is Reserved')
				else:
					payroot = Tk()
					payroot.title("Test")
					payroot.minsize(height=236,width=302)
					payroot.configure(bg='White')
					#global pmethod
					cur.execute("select price from roomd where rn = (?)",(roomn.get(),))
					rp = cur.fetchone()
					print (rp)
					amtpd = str(int(rp[0])*int(nod.get()))
					Label(payroot,text='Select an option to pay '+str(int(rp[0])*int(nod.get())),font='msserif 14 bold',bg='White').place(x=0,y=0)
					Frame(payroot,height=4,width=300,bg='cyan4').place(x=0,y=39)
					Radiobutton(payroot,text='Cash  ',bg='White',variable=pmethod,value=1,font='helvetica 15',width=5).place(x=0,y=43+10)
					Radiobutton(payroot,text='Card   ',bg='White',variable=pmethod,value=2,font='helvetica 15',width=5).place(x=0,y=80+10)
					Radiobutton(payroot,text='UPI     ',bg='White',variable=pmethod,value=3,font='helvetica 15',width=5).place(x=0,y=115+10)
					Radiobutton(payroot,text='Paytm ',bg='White',variable=pmethod,value=4,font='helvetica 15',width=5).place(x=0,y=150+10)
					def f():
						if pmethod != '':
							print (pmethod.get())
							print ('pmethod value')
							cur.execute("select id from paymentsf order by id desc")
							x = cur.fetchone()
							cid = int(x[0])
							cid+=1
							cur.execute("insert into paymentsf values(?,?,?,?,?,?,?,?,?,?,?,?)",(cid,fn.get(),ln.get(),cn.get(),em.get(),roomn.get(),str(now.strftime("%d")),str(now.strftime("%b")),str(now.strftime("%Y")),str(now.strftime("%H:%M")),str(pmethod.get()),amtpd))
							cur.execute("update roomd set rstatus='Reserved' where rn = ? ",(roomn.get(),))
							messagebox.showinfo("Successful","Room Booked successfully")
							con.commit()
							ask = messagebox.askyesno("Successful","Payment Successful\nDo you want to print reciept ?")
							if ask == 'yes':
								def createfile():
									fl = open("reciept.txt","w")
									fl.write("reciept will come here")
							reserve()
							payroot.destroy()
						else :
							messagebox.showwarning("Not selected","Please Select the payment method")
					Button(payroot,text='Pay',font='msserif 12',bg='Green',fg='White',width=28,command=f).place(x=0,y=200)
					Label(payroot,text='Your unique payment id :',font='msserif',bg='White')#.place(x=0,y=25)
		def unreserve():
			if (roomn.get() == 'Enter Room Number') or (roomn.get()==''):
				messagebox.showerror('Entries not filled','Kindly Enter room Number')
			else :
				cur.execute("update roomd set rstatus='Unreserved' where rn = ? ",(roomn.get(),))
				messagebox.showinfo("Successful","Room Unreserved successfully")
				reserve()
				con.commit()
		#--------------------------------------------------------right side---------------------------------------------------
		Label(b_frame,text='Filter',font='msserif 20',bg='gray93').place(x=850,y=0)
		nbb = IntVar()
		acb = IntVar()
		tvb = IntVar()
		wifib = IntVar()
		style = ttk.Style()
		style.map('TCombobox', fieldbackground=[('readonly','white')])
		Label(b_frame,text='Bed(s) :',bg='gray93',font='17').place(x=730,y=50)
		nb = ttk.Combobox(b_frame,values=['please select...','1','2','3'],state='readonly',width=22)
		nb.place(x=830,y=50)
		nb.current(0)
		Label(b_frame,text='AC :',font='17',bg='gray93').place(x=732,y=75)
		ac = ttk.Combobox(b_frame,values=['please select...','Yes','No'],state='readonly',width=22)
		ac.place(x=830,y=75)
		ac.current(0)
		Label(b_frame,text='TV :',font='17',bg='gray93').place(x=732,y=100)
		tv = ttk.Combobox(b_frame,values=['please select...','Yes','No'],state='readonly',width=22)
		tv.place(x=830,y=100)
		tv.current(0)
		Label(b_frame,text='Wifi :',font='17',bg='gray93').place(x=732,y=125)
		wifi = ttk.Combobox(b_frame,values=['please select...','Yes','No'],state='readonly',width=22)
		wifi.place(x=830,y=125)
		wifi.current(0)
		listofrooms = Listbox(b_frame,height=6,width=36)
		listofrooms.place(x=735,y=190)
		listofrooms.insert(END,'Rooms of Your Choice will appear Here')
		listofrooms.insert(END,'once you apply filter')
		def findrooms():
			cur.execute('select rn,price,rstatus from roomd where beds = ? and ac = ? and tv = ? and internet = ? order by price asc',((nb.get()),ac.get(),tv.get(),wifi.get()) )
			x = cur.fetchall()
			listofrooms.delete(0,END)
			if x == []:
				listofrooms.insert(END,'No Matching Found')
			for i in x :
				listofrooms.insert(END,'Room Number '+str(i[0])+' - Price - '+str(i[1]))
		Res = Button(b_frame,text='Reserve',bg='white',fg='cyan4',font='timenewroman 11',activebackground='green',command=booking).place(x=235,y=270)
		unres = Button(b_frame,text='Unreserve',bg='white',fg='cyan4',font='timenewroman 11',activebackground='green',command=unreserve).place(x=327,y=270)
		findrooms = Button(b_frame,text='Find Rooms',bg='white',fg='cyan4',font='timenewroman 9',activebackground='green',command = findrooms).place(x=830,y=155)
		scrollbar = Scrollbar(b_frame, orient="vertical")
		scrollbar.config(command=listofrooms.yview)
		scrollbar.place(x=1014,y=191,height=111)
		listofrooms.config(yscrollcommand=scrollbar.set)
		b_frame.place(x=0,y=120+6+20+60+11)
		b_frame.pack_propagate(False)
		b_frame.tkraise()
		nl = Label(b_frame,text='Made by Avengers',fg='black',bg='gray91',font='msserif 8')
		nl.place(x=955,y=310)
		nl.tkraise()
	#-------------login module----------------------------------------------------------------------------------------------------------------------
	def login():
		q = messagebox.askyesno("Exit","Do you really want to exit ?")
		if(q):
			root.destroy()
	#---------------2nd top frame-----------------------------------------------------------------------------------------------------------------
	sl_frame = Frame(root,height=130,width=1080,bg='white')
	sl_frame.place(x=0,y=70+6)
	path = "images/rooms.png"
	img = ImageTk.PhotoImage(Image.open(path))
	b1 = Button(sl_frame,image=img,text='b1',bg='white',width=180,command=rooms)
	b1.image = img
	b1.place(x=180,y=0)
	path2 = "images/hotelstatus.png"
	img1 = ImageTk.PhotoImage(Image.open(path2))
	b2 = Button(sl_frame,image=img1,text='b2',bg='white',width=180,command=hotel_status)
	b2.image = img1
	b2.place(x=0,y=0)
	path3='images/guests.png'
	img3 = ImageTk.PhotoImage(Image.open(path3))
	b3 = Button(sl_frame,image=img3,text='b2',bg='white',width=180,command=staff)
	b3.image = img3
	b3.place(x=180*4,y=0)
	path4='images/payments.png'
	img4 = ImageTk.PhotoImage(Image.open(path4))
	b4 = Button(sl_frame,image=img4,text='b2',bg='white',width=180,command = payments)
	b4.image = img4
	b4.place(x=180*3,y=0)
	path5='images/logout.png'
	img5 = ImageTk.PhotoImage(Image.open(path5))
	b5 = Button(sl_frame,image=img5,text='b2',bg='white',width=180,height=100,command=login)
	b5.image = img5
	b5.place(x=180*5,y=0)
	path6='images/Bookroom.png'
	img6 = ImageTk.PhotoImage(Image.open(path6))
	b6 = Button(sl_frame,image=img6,text='b2',bg='white',width=180,height=100,command=reserve)
	b6.image = img6
	b6.place(x=180*2,y=0)
	Label(sl_frame,text='Hotel Status',font='msserif 13',bg='white').place(x=35,y=106)
	Label(sl_frame,text='Hotels',font='msserif 13',bg='white').place(x=248,y=106)
	Label(sl_frame,text='Time',font='msserif 13',bg='white').place(x=417,y=106)
	Label(sl_frame,text='Contacts',font='msserif 13',bg='white').place(x=774,y=106)
	Label(sl_frame,text='Test',font='msserif 13',bg='white').place(x=615,y=106)
	Label(sl_frame,text='Exit',font='msserif 13',bg='white').place(x=968,y=106)
	sl_frame.pack_propagate(False)
	#-------------------extra frame------------------------------------------------------------------------------------------------------------------
	redf = Frame(root,height=6,width=1080,bg='lightsteelblue3')
	redf.place(x=0,y=70)
	redf1 = Frame(root,height=40,width=1080,bg='lightsteelblue3')
	redf1.place(x=0,y=210)
	#-------------------------------------------------------------------------------------------------------------------------------------------------
	reserve()
	datetime()
	mainloop()
def call_mainroot():
	sroot.destroy()
	mainroot()
sroot.after(3000,call_mainroot)
mainloop()