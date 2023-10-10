
import tkinter as tk
import pandas as pd
from tkinter import filedialog, ttk
from sklearn import preprocessing
from sklearn.linear_model  import LinearRegression , Ridge ,ElasticNet ,Lasso
from sklearn.preprocessing import LabelEncoder ,PolynomialFeatures ,MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import re
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import xgboost as xgb
from joblib import dump, load


import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np

import tkinter as tk
import pandas as pd
from tkinter import filedialog, ttk
from sklearn import preprocessing
from sklearn.linear_model  import LinearRegression , Ridge ,ElasticNet ,Lasso
from sklearn.preprocessing import LabelEncoder ,PolynomialFeatures ,MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import re
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import xgboost as xgb
from joblib import dump, load
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time


# # Handle nulls




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


# ## Feature Encoding 


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


# # Load Data 


train_data = pd.read_csv("D:\\Collage\\Sem 6\\machine learning\\Project\\MS2\\hotel-classification-dataset.csv")
train_data['Reviewer_Score'] = train_data['Reviewer_Score'].map({'Low_Reviewer_Score': 0,'Intermediate_Reviewer_Score': 1, 'High_Reviewer_Score': 2})

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


# # Preprocessing 


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
            data = data.iloc[:, :-1]
            # y_test = data['Reviewer_Score'] 
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
    

#test = pd.read_csv('test.csv')

X_train,fillvalues ,encoders = pre(X_train,0,0,1,0)
X_test = pre(X_test,fillvalues,encoders,0,0)
#test = pre(test,fillvalues,encoders,0,1)
        


# ## Models 


print(X_train)



# X_train = X_train.drop(['lat','Total_Number_of_Reviews_Reviewer_Has_Given','lng','year','day','Reviewer_Nationality'],axis = 1)
# X_test = X_test.drop(['lat','Total_Number_of_Reviews_Reviewer_Has_Given','lng','year','day','Reviewer_Nationality'],axis = 1)
##model selection
# from sklearn.feature_selection import RFE

# classifiers = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier( max_depth=3, learning_rate=0.1, n_estimators=100)]

# # Train and evaluate each classifier
# for clf in classifiers:
#     model = clf
#     rfe = RFE(model, n_features_to_select=10)
#     print(type(clf).__name__)
#     rfe.fit(X_train, y_train)
#     features=X_train.columns[rfe.support_]
#     print("features " , features)

# lr = LogisticRegression()

# # Define the hyperparameters for grid search
# param_grid = {
#     'C': [0.1, 1, 10],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga']
# }

# # Perform grid search using cross-validation
# grid_search = GridSearchCV(lr, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters and the corresponding accuracy score
# print("Best hyperparameters:", grid_search.best_params_)
# y_pred = grid_search.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy:", acc)

models = [

    ('Random Forest', RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [10],
        'min_samples_split': [2, 4, 6]
    }),
    
   
    ('Logistic Regression', LogisticRegression(), {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }),
    ('Decision Tree', DecisionTreeClassifier(), {
        'max_depth': [6,10],
        'min_samples_split': [2, 4, 6]
    })
    
]

# Perform grid search for each model using cross-validation
# for name, model, params in models:
#     grid_search = GridSearchCV(model, params, cv=5)
#     grid_search.fit(X_train, y_train)
    
#     # Print the best hyperparameters and the corresponding accuracy score
#     print("Model:", name)
#     print("Best hyperparameters:", grid_search.best_params_)
#     y_pred = grid_search.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print("Accuracy:", acc)


# # models regression  



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
    rf_classifier = DecisionTreeClassifier(max_depth=10)
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


