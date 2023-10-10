import re
import ast
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures


def handle_duplicated(data):
    print(data.duplicated().sum())
    data = data.drop_duplicates()
    print(len(data))
    return data


# Preprocessing Functions
# Get Date
def change_date(data, col):
    data[col] = pd.to_datetime(data[col])
    data['year'] = data[col].dt.year.astype(float)
    data['month'] = data[col].dt.month.astype(float)
    data['day'] = data[col].dt.day.astype(float)
    data = data.drop(['Review_Date'], axis=1)
    return data

# Get Address


def get_address(data):
    col = data['Hotel_Address']
    x = data['Hotel_Address'].str[-14:]
    data['Hotel_Address'] = data['Hotel_Address'].apply(
        lambda x: x.split()[-1])
    data['Hotel_Address'] = data['Hotel_Address'].apply(
        lambda x: 'United Kingdom' if x == 'Kingdom' else x)
    data['Hotel_Address'] = data['Hotel_Address'].apply(
        lambda x: 'United States' if x == 'States' else x)
    return data

# Get Days


def get_daysNo(data):
    pattern = r'\d+'
    data['days_since_review'] = data['days_since_review'].apply(
        lambda x: int(re.findall(pattern, x)[0]))
    return data

# Get Tags


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

# Feature Encoding


def Feature_Encoder(data, label, encoders=None):
    if encoders is None:
        encoders = {}

    cols = ('Hotel_Name', 'Reviewer_Nationality', 'Room', 'Trip',
            'Nights', 'Hotel_Address', 'Positive_Review', 'Negative_Review')

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
                    unseen_values = np.setdiff1d(new_values.astype(
                        str), encoders[col].classes_.astype(str))
                    if len(unseen_values) > 0:
                        encoders[col].classes_ = np.append(
                            encoders[col].classes_, unseen_values)
                    data[col] = encoders[col].transform(data[col])
                else:
                    print(f"LabelEncoder not found for column '{col}'.")
            else:
                print(f"Column '{col}' not found in the test dataset.")

    return data, encoders

# Scaling Data


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


# Handle nulls
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


def pre(data, fillvalues, encoder, label, test):
    if label:
        data = change_date(data, 'Review_Date')
        print('done date')
        print('done nulls')
        data = get_address(data)
        print('done add')
        data = get_daysNo(data)
        print('done day')
        data = get_tags(data)
        print('done tags')
        data, fill_values = handle_nulls_in_train_data(data)
        print('handel nulls done')
        print(data.info())
        data, encoders = Feature_Encoder(data, label=True)
        print('encoding done in train')
        print(data.info())
        data = scaler_fit_transform(data)

        return data, fill_values, encoders

    else:
        if (test):
            data['Reviewer_Score'] = data['Reviewer_Score'].map(
                {'Low_Reviewer_Score': 0, 'Intermediate_Reviewer_Score': 1, 'High_Reviewer_Score': 2})
            y_test = data['Reviewer_Score']
            data = data.iloc[:, :-1]
            data = change_date(data, 'Review_Date')
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
            data, _ = Feature_Encoder(data, label=False, encoders=encoder)
            print('done encoder in test')
            print(data.info())
            return data, y_test
        else:
            data = change_date(data, 'Review_Date')
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
            data, _ = Feature_Encoder(data, label=False, encoders=encoder)
            print('done encoder in test')
            print(data.info())
            data = scaler_transform(data)
            return data
