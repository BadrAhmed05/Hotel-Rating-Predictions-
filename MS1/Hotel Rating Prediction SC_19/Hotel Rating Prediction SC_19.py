import tkinter as tk
import pandas as pd
from tkinter import filedialog, ttk
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


def handle_duplicated(data):
    print(data.duplicated().sum())
    data = data.drop_duplicates()
    print(len(data))
    return data

def handle_nulls_in_test_data(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(X_train[column].mode()[0], inplace=True)
        else:
            data[column].fillna(X_train[column].median(), inplace=True)
    return data

def get_address(data):
    col =data['Hotel_Address']
    x=data['Hotel_Address'].str[-14:]
    data['Hotel_Address'] = data['Hotel_Address'].apply(lambda x: x.split()[-1])
    data['Hotel_Address'] =data['Hotel_Address'].apply(lambda x: 'United Kingdom' if x == 'Kingdom' else x)
    data['Hotel_Address'] =data['Hotel_Address'].apply(lambda x: 'United States' if x == 'States' else x)
    return data

def change_date(data, col):
    data[col] = pd.to_datetime(data[col])
    data['year'] = data[col].dt.year
    data['month'] = data[col].dt.month
    data['day'] = data[col].dt.day
    return data

def Feature_Encoder(X_train, X_test):
    label_encoder = LabelEncoder()
    cols = ('Hotel_Name', 'Reviewer_Nationality', 'Negative_Review', 'Positive_Review','Room','Trip','Nights')

    # combine the training and test data
    combined_data = pd.concat([X_train, X_test], axis=0)

    # fit the label encoder to the combined data
    for column in cols:
        label_encoder.fit(combined_data[column])
        X_train[column] = label_encoder.transform(X_train[column])
        X_test[column] = label_encoder.transform(X_test[column])
  
    return X_train, X_test 

def get_daysNo(data):
    pattern = r'\d+'

    # apply the regular expression pattern to the column and extract the numeric values
    data['days_since_review'] = data['days_since_review'].apply(lambda x: int(re.findall(pattern, x)[0]))
    return data
def min_max_scale(train_data, test_data):
    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # fit and transform the training data
    train_data_scaled = scaler.fit_transform(train_data)

    # transform the test data using the same scaler
    test_data_scaled = scaler.transform(test_data)

    # convert the scaled data back to data frames
    train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.columns)
    test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns)

    return train_data_scaled, test_data_scaled

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

data = pd.read_csv("D:\\Collage\\Sem 6\\machine learning\\Project\\MS1\\hotel-regression-dataset.csv")

data.info()

data = handle_duplicated(data)

X = data.iloc[:, :-1]

Y = data['Reviewer_Score']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(len(X_train))

print (X_train.isna().sum())

X_train = handle_nulls_in_test_data(X_train)
print(len(X_train))

print(X_train)

X_train = get_address(X_train)
print(X_train.head(10))

X_train = handle_duplicated(X_train)
print(X_train)

X_train = change_date(X_train,'Review_Date')
print(X_train.info())

print(X_train.info())

X_train = get_daysNo(X_train)
print(X_train.info())

X_train = X_train.drop(['Hotel_Address','Review_Date'],axis= 1 )
print(X_train.info())

X_test = X_test.drop(['Hotel_Address'],axis= 1 )

X_test = change_date(X_test,'Review_Date')

X_train = get_tags(X_train)
X_test = get_tags(X_test)

X_test = X_test.drop(['Review_Date'],axis = 1)
print(X_test.info())

X_test = handle_nulls_in_test_data(X_test)
print(X_test.info())

print(X_test.info())

X_train ,X_test = Feature_Encoder(X_train,X_test)
print(X_test)

X_test = get_daysNo(X_test)
print(X_test.info())

print(X_test.duplicated().sum())

X_train,X_test = min_max_scale(X_train,X_test)
print(X_test.info())

print(X_train.info())
plt.figure(figsize=(20,12))

sns.heatmap(data=pd.concat([X_train,y_train],axis=1).corr(),annot=True)

plt.show()

from sklearn.ensemble import RandomForestRegressor
import pickle
import time


def regression_models(X_train, y_train, X_test, y_test):
    
    
    

    def LR():
        start = time.time()
        model = LinearRegression().fit(X_train, y_train)
        with open('linear_regression_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        end = time.time()
        training_time = end - start
        start1 = time.time()
        with open('linear_regression_model.pkl', 'rb') as file:
            lr_loaded = pickle.load(file)
        y_pred = lr_loaded.predict(X_test)
        y_train_pred=lr_loaded.predict(X_train)
        mse = mean_squared_error(y_test, y_pred)
        mse_train=mean_squared_error(y_train, y_train_pred)
        end1 = time.time()
        test_time = end1 - start1
        # print("MSE for x_test = ", mse , '%')
        # print("MSE for x_train = ", mse_train , '%')
        # print('Training Time = ', training_time, 's')
        # print('Test Time = ', test_time, 's')

        return mse,mse_train, training_time, test_time
    def Rd():
        start = time.time()
        model = Ridge(alpha=0.01).fit(X_train, y_train)
        with open('Ridge_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        end = time.time()
        training_time = end - start
        start1 = time.time()
        with open('Ridge_model.pkl', 'rb') as file:
            lr_loaded = pickle.load(file)
        y_pred = lr_loaded.predict(X_test)
        y_train_pred=lr_loaded.predict(X_train)
        mse = mean_squared_error(y_test, y_pred)
        mse_train=mean_squared_error(y_train, y_train_pred)
        end1 = time.time()
        test_time = end1 - start1
        # print("MSE for x_test = ", mse , '%')
        # print("MSE for x_train = ", mse_train , '%')
        # print('Training Time = ', training_time, 's')
        # print('Test Time = ', test_time, 's')

        return mse,mse_train, training_time, test_time
   
    def poly():
        start = time.time()
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression().fit(X_poly, y_train)
        with open('poly_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        end = time.time()
        training_time = end - start
        
        start1 = time.time()
        with open('poly_model.pkl', 'rb') as file:
            model_loaded = pickle.load(file)
        y_pred = model_loaded.predict(poly.transform(X_test))
        y_train_pred = model_loaded.predict(poly.transform(X_train))
        mse = mean_squared_error(y_test, y_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        end1 = time.time()
        test_time = end1 - start1
        
        return mse, mse_train, training_time, test_time
    def lasso():
        start = time.time()
        model = Lasso(alpha=0.01).fit(X_train, y_train)
        with open('lasso_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        end = time.time()
        training_time = end - start
        start1 = time.time()
        with open('lasso_model.pkl', 'rb') as file:
            lr_loaded = pickle.load(file)
        y_pred = lr_loaded.predict(X_test)
        y_train_pred=lr_loaded.predict(X_train)
        mse = mean_squared_error(y_test, y_pred)
        mse_train=mean_squared_error(y_train, y_train_pred)
        end1 = time.time()
        test_time = end1 - start1
        # print("MSE for x_test = ", mse , '%')
        # print("MSE for x_train = ", mse_train , '%')
        # print('Training Time = ', training_time, 's')
        # print('Test Time = ', test_time, 's')

        return mse,mse_train, training_time, test_time
    def Dt():
        start = time.time()
        model = DecisionTreeRegressor(random_state=50, max_depth=10).fit(X_train, y_train)
        with open('dtr_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        end = time.time()
        training_time = end - start
        start1 = time.time()
        with open('dtr_model.pkl', 'rb') as file:
            lr_loaded = pickle.load(file)
        y_pred = lr_loaded.predict(X_test)
        y_train_pred=lr_loaded.predict(X_train)
        mse = mean_squared_error(y_test, y_pred)
        mse_train=mean_squared_error(y_train, y_train_pred)
        end1 = time.time()
        test_time = end1 - start1
        # print("MSE for x_test = ", mse , '%')
        # print("MSE for x_train = ", mse_train , '%')
        # print('Training Time = ', training_time, 's')
        # print('Test Time = ', test_time, 's')

        return mse,mse_train, training_time, test_time
    def RF():
        start = time.time()
        model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=1).fit(X_train, y_train)
        with open('rf_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        end = time.time()
        training_time = end - start
        start1 = time.time()
        with open('rf_model.pkl', 'rb') as file:
            lr_loaded = pickle.load(file)
        y_pred = lr_loaded.predict(X_test)
        y_train_pred=lr_loaded.predict(X_train)
        mse = mean_squared_error(y_test, y_pred)
        mse_train=mean_squared_error(y_train, y_train_pred)
        end1 = time.time()
        test_time = end1 - start1
        print("MSE for x_test = ", mse , '%')
        print("MSE for x_train = ", mse_train , '%')
        print('Training Time = ', training_time, 's')
        print('Test Time = ', test_time, 's')

        return mse,mse_train, training_time, test_time
           


    return {
        "LR": LR(),
        "Rd": Rd(),
        "poly": poly(),
        "lasso": lasso(),
        "Decision": Dt()
         # "Random": RF()
    }

# Call the regression_models function
results = regression_models(X_train, y_train, X_test, y_test)
# Extract the MSE values for each model
# mse_lr = results["LR"]
mse_lr_test,mse_lr_train,trainning_time_lr,testing_time_lr=results["LR"]
mse_rd_test,mse_rd_train,trainning_time_rd,testing_time_rd = results["Rd"]
mse_poly_test,mse_poly_train,trainning_time_poly,testing_time_poly = results["poly"]
mse_lasso_test,mse_lasso_train,trainning_time_lasso,testing_time_lasso = results["Rd"]
mse_Decision_test,mse_Decision_train,trainning_time_Decision,testing_time_Decision= results["Decision"]




# Print the MSE for each model for test data
print("-----------------------------------")
print("Linear Regression MSE for test data:", mse_lr_test)
print("-----------------------------------")
print("Ridge Regression MSE for test data:", mse_rd_test)
print("-----------------------------------")
print("Polynomial Regression MSE for test data:", mse_poly_test)
print("-----------------------------------")
print("Lasso Regression MSE for test data:",mse_lasso_test)
print("-----------------------------------")
print("Decision tree MSE for test data:", mse_Decision_test)
print("-----------------------------------")
# Print the MSE for each model for train data
print("-----------------------------------")
print("Linear Regression MSE for train data:", mse_lr_train)
print("-----------------------------------")
print("Ridge Regression MSE for train data:", mse_rd_train)
print("-----------------------------------")
print("Polynomial Regression MSE for train data:", mse_poly_train)
print("-----------------------------------")
print("Lasso Regression MSE for train data:",mse_lasso_train)
print("-----------------------------------")
print("Decision tree MSE for train data:", mse_Decision_train)
print("-----------------------------------")
# Print the time for each model for training data
print("-----------------------------------")
print(f"Linear Regression Training Time:  (Time: {trainning_time_lr:.2f} seconds)")
print("-----------------------------------")
print(f"Ridge Regression Training Time:  (Time: {trainning_time_rd:.2f} seconds)")
print("-----------------------------------")
print(f"Polynomial Regression Training Time:  (Time: {trainning_time_poly:.2f} seconds)")
print("-----------------------------------")
print(f"Lasso Regression Training Time:  (Time: {trainning_time_lasso:.2f} seconds)")
print("-----------------------------------")
print(f"Decision Regression Training Time:  (Time: {trainning_time_Decision:.2f} seconds)")
print("-----------------------------------")
# Print the time for each model for test data
print("-----------------------------------")
print(f"Linear Regression Testing Time:  (Time: {testing_time_lr:.2f} seconds)")
print("-----------------------------------")
print(f"Ridge Regression Testing Time:  (Time: {testing_time_rd:.2f} seconds)")
print("-----------------------------------")
print(f"Polynomial Regression Testing Time:  (Time: {testing_time_poly:.2f} seconds)")
print("-----------------------------------")
print(f"Lasso Regression Testing Time:  (Time: {testing_time_lasso:.2f} seconds)")
print("-----------------------------------")
print(f"Decision Regression Testing Time:  (Time: {testing_time_Decision:.2f} seconds)")
print("-----------------------------------")
import matplotlib.pyplot as plt



# Define the models and their corresponding MSE values for test data
models = ['Linear Regression', 'Ridge Regression', 'Polynomial Regression', 'Lasso Regression', 'Decision Tree']
mse_values = [mse_lr_test, mse_rd_test, mse_poly_test, mse_lasso_test, mse_Decision_test]

# Set the width of the bars
bar_width = 0.35

# Create a numpy array for the x-axis positions of the bars
x_pos = range(len(models))

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the bars
bars = ax.bar(x_pos, mse_values, bar_width)

# Add labels, title, and legend
ax.set_xlabel('Regression Models')
ax.set_ylabel('Mean Squared Error For Test Data')
ax.set_title('Comparison of Regression Models')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

# Show the plot
plt.show()

# Define the models and their corresponding MSE values for train data
models = ['Linear Regression', 'Ridge Regression', 'Polynomial Regression', 'Lasso Regression', 'Decision Tree']
mse_values = [mse_lr_train, mse_rd_train, mse_poly_train, mse_lasso_train, mse_Decision_train]

# Set the width of the bars
bar_width = 0.35

# Create a numpy array for the x-axis positions of the bars
x_pos = range(len(models))

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the bars
bars = ax.bar(x_pos, mse_values, bar_width)

# Add labels, title, and legend
ax.set_xlabel('Regression Models')
ax.set_ylabel('Mean Squared Error For Train Data')
ax.set_title('Comparison of Regression Models')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

# Show the plot
plt.show()

##########################################
# Define the time of test data and train data 
n_groups = 5
means_frank = (trainning_time_lr, trainning_time_rd, trainning_time_poly, trainning_time_lasso, trainning_time_Decision)
means_guido = (testing_time_lr, testing_time_rd, testing_time_poly, testing_time_lasso, testing_time_Decision)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Training Time')

rects2 = ax.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Testing Time')

ax.set_xlabel('Model')
ax.set_ylabel('Time (s)')
ax.set_title('Training and Testing Time by Model')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('LinearR', 'Ridge', 'Poly', 'Lasso', 'Decision'))
ax.legend()

fig.tight_layout()
plt.show()