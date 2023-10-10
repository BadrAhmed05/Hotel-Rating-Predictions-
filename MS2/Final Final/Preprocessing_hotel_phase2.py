import pandas as pd
from Models import *
import matplotlib.pyplot as plt 
from Preprocessing_functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Load Data 
train_data = pd.read_csv("D:\\Collage\\Sem 6\\machine learning\\Project\\MS2\\hotel-classification-dataset.csv")
train_data['Reviewer_Score'] = train_data['Reviewer_Score'].map({'Low_Reviewer_Score': 0,'Intermediate_Reviewer_Score': 1, 'High_Reviewer_Score': 2})
train_data = handle_duplicated(train_data)


def removeoutliers(data):
    for i, col in enumerate(train_data.select_dtypes(include=['number']).columns):
        # axs[i, 0].scatter(data.index, data[col])
        # axs[i, 0].set_xlabel('Index')
        # axs[i, 0].set_ylabel(col)
    
        Q1 = train_data[col].quantile(0.25)
        Q3 = train_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        outliers = data[(train_data[col] < lower_bound) | (train_data[col] > upper_bound)]
        data = data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
        return train_data
        
train_data = removeoutliers(train_data)    

X = train_data.iloc[:, :-1]
Y = train_data['Reviewer_Score']

# Data Spliting 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


test = pd.read_csv("E:\\Test Samples\\hotel-tas-test-classification.csv")

# Preprocessing     
X_train,fillvalues ,encoders = pre(X_train,0,0,1,0)
X_test = pre(X_test,fillvalues,encoders,0,0)
test, yt = pre(test,fillvalues,encoders,0,1)
X_test = test
y_test = yt


# Loadmodel(X_test, y_test)


# print("LENX:", len(test))
# print("LENY:", len(yt))

# X_train = X_train.drop(['lat','Total_Number_of_Reviews_Reviewer_Has_Given','lng','year','day','Reviewer_Nationality'],axis = 1)
# X_test = X_test.drop(['lat','Total_Number_of_Reviews_Reviewer_Has_Given','lng','year','day','Reviewer_Nationality'],axis = 1)


# Models 
# model selection
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

# models = [
#     ('Random Forest', RandomForestClassifier(), {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [10],
#         'min_samples_split': [2, 4, 6]
#     }),

#     ('Logistic Regression', LogisticRegression(), {
#         'C': [0.1, 1, 10],
#         'penalty': ['l1', 'l2'],
#         'solver': ['liblinear', 'saga']
#     }),

#     ('Decision Tree', DecisionTreeClassifier(), {
#         'max_depth': [6,10],
#         'min_samples_split': [2, 4, 6]
#     })]

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
    
# calling models 
accuracy_rf, training_time_rf, test_time_rf = Random_Forest_model(X_train,y_train,X_test,y_test)
accuracy_gb, training_time_gb, test_time_gb = Gradient_Boosting_model(X_train,y_train,X_test,y_test)
accuracy_lr, training_time_lr, test_time_lr = Logistic_Regression_model(X_train,y_train,X_test,y_test)
accuracy_dt, training_time_dt, test_time_dt = Decision_Tree_model(X_train,y_train,X_test,y_test)
accuracy_xg, training_time_xg, test_time_xg = xgb_model(X_train,y_train,X_test,y_test)
# accuracy_svm, training_time_svm, test_time_svm = SVM_model(X_train,y_train,X_test,y_test)

# Ploting 
# Acurecies Comparison
print("=====================================")
print("Logistic Regression Accuracy = {:.2%}".format(accuracy_dt))
print("=====================================")
# print("Gradient Boosting Accuracy = {:.2%}".format(accuracy_gb))
# print("=====================================")
print("Decision Tree Accuracy = {:.2%}".format(accuracy_dt))
print("=====================================")
print("XGB Accuracy = {:.2%}".format(accuracy_xg))
print("=====================================")
print("Random Forest Accuracy {:.2%}".format(accuracy_dt))
print("=====================================")
# print("SVM Accuracy = {:.2%}".format(accuracy_svm))
# print("=====================================")

# Print the Training time for each model
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
# print(f"SVM Training Time:  (Time: {training_time_svm:.2f} seconds)")
# print("=====================================")

# Print the Test time for each model
# print("=====================================")
print(f"Logistic Regression Testing Time:  (Time: {test_time_lr:.2f} seconds)")
print("=====================================")
# print(f"SVM Testing Time:  (Time: {test_time_svm:.2f} seconds)")
# print("=====================================")

# print(f"Gradient Boosting Testing Time:  (Time: {test_time_gb:.2f} seconds)")
# print("=====================================")
print(f"Decision Tree Testing Time:  (Time: {test_time_dt:.2f} seconds)")
print("=====================================")
print(f"XGB Accuracy Testing Time:  (Time: {test_time_xg:.2f} seconds)")
print("=====================================")
print(f"Random Forest Testing Time:  (Time: {test_time_rf:.2f} seconds)")
print("=====================================")
# print(f"SVM Testing Time:  (Time: {test_time_svm:.2f} seconds)")
# print("=====================================")


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