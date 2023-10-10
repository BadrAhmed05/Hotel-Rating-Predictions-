import time
import pickle
import numpy as np 
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Models

# Logistic_Regression
def Logistic_Regression_model(X_train, Y_train, X_test, Y_test):
    start = time.time()
    model = LogisticRegression(C=1, max_iter=1000).fit(X_train, Y_train)
    # with open('logistic_regression_model.pkl', 'wb') as file:
    #     pickle.dump(model, file)
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

# Decision_Tree
def Decision_Tree_model(X_train, Y_train, X_test, Y_test):
    start_time = time.time()
    rf_classifier = DecisionTreeClassifier(max_depth=10)
    Y_train = np.ravel(Y_train)
    rf_classifier.fit(X_train, Y_train)
    # with open('decision_tree_model.pkl', 'wb') as file:
    #     pickle.dump(rf_classifier, file)
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

# Random_Forest
def Random_Forest_model(X_train,Y_train,X_test,Y_test):
    start_time = time.time()
    rf_classifier = RandomForestClassifier(max_depth=10)
    Y_train = np.ravel(Y_train)
    rf_classifier.fit(X_train, Y_train)
    # with open('random_forest_model.pkl', 'wb') as file:
    #     pickle.dump(rf_classifier, file)
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

# Gradient_Boosting
def Gradient_Boosting_model(X_train,Y_train,X_test,Y_test):
    start_time = time.time()
    # Create an instance of the GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier()

    # Fit the model to the training data
    Y_train = np.ravel(Y_train)
    gb_classifier.fit(X_train, Y_train)
    # with open('Gradient_Boosting_model.pkl', 'wb') as file:
    #     pickle.dump(gb_classifier, file)
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

# XGB
def xgb_model(X_train,Y_train,X_test,Y_test):
    start_time = time.time()
    # Create an instance of the GradientBoostingClassifier
    gb_classifier = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100)

    # Fit the model to the training data
    Y_train = np.ravel(Y_train)
    gb_classifier.fit(X_train, Y_train)
    # with open('xgb_model.pkl', 'wb') as file:
    #     pickle.dump(gb_classifier, file)
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

def SVM_model(X_train,Y_train,X_test,Y_test):
    start_time = time.time()
    # Create an instance of the GradientBoostingClassifier
    svm = SVC(kernel='linear', C=0.1)

    # Fit the model to the training data
    Y_train = np.ravel(Y_train)
    svm.fit(X_train, Y_train)
    # with open('svm_model.pkl', 'wb') as file:
    #     pickle.dump(svm, file)
    end_time = time.time()
    training_time = end_time - start_time

    # Predict on the test data
    start_time = time.time()
    with open('svm_model.pkl', 'rb') as file:
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
    y_pred = loaded.predict(X_test) 
    end_time = time.time() 
    test_time = end_time - start_time
    accuracy = accuracy_score(Y_test, y_pred) 
    print("Accuracy = ", accuracy * 100, '%')




    
    
    