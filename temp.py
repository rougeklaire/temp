import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("PATH_TO.CSV_FILE")

#drop features not needed in the model
droppable_features = ["state", "driver_race_raw", "stop_date"]
data = data.drop(columns = droppable_features)

# Handle missing values, dropping each row containing missing values from data
data = data.dropna()

#debugging
""" def print_all_unique_pre():
    for column in data.columns:
        unique_values = data[column].unique()
        print(f"{column}: {unique_values}")
print_all_unique_pre() """

# Convert categorical features to numerical using one-hot encoding
categorical_features = ['driver_gender', 'driver_race', 'violation', 'search_type', 'contraband_found', "search_basis", "district"]  #officer_id?
one_hot_encoder = OneHotEncoder(drop = "first")
one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_features])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(categorical_features))

#debugging
""" print("-----START OF CHECK-------")
def print_all_unique():
    for column in label_encoded_df.columns:
        unique_values = label_encoded_df[column].unique()
        print(f"{column}: {unique_values}")
print_all_unique()
print("------END OF CHECK-----")

print("Index of 'data':", data.index)
print("Index of 'label_encoded_df':", label_encoded_df.index) """


# Drop original categorical columns and add encoded columns
data = data.drop(columns = categorical_features)

#debugging
""" print("---- START OF DEBUG BEFORE CONCATENATING ----- /n")
def print_all_unique():
    for column in data.columns:
        unique_values = data[column].unique()
        print(f"{column}: {unique_values}")
print_all_unique()
print("/n ---- END OF DEBUG BEFORE CONCATENATING ----- /n") """

data.reset_index(drop=True, inplace=True)

data = pd.concat([data, one_hot_encoded_df], axis=1)

""" print("---- START OF DEBUG AFTER CONCATENATING ----- /n")

def print_all_unique():
    for column in data.columns:
        unique_values = data[column].unique()
        print(f"{column}: {unique_values}")
print_all_unique()

print("/n---- END OF DEBUG BEFORE CONCATENATING ----- /n") """


# Normalize numerical features
numerical_features = ['driver_age']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Split the data into training and testing sets and implement logistic regression
X = data.drop(columns=['stop_outcome'])
#X = X.dropna()

""" #debugging
def print_all_unique():
    for column in X.columns:
        unique_values = X[column].unique()
        print(f"{column}: {unique_values}")
print_all_unique() """


label_encoder = LabelEncoder()
y = data['stop_outcome']
y = label_encoder.fit_transform(data["stop_outcome"])

#if necessary reformat "y" to a pandas dataframe
#y = pd.DataFrame(data = y, columns = ["stop_outcome"])

#y = y.dropna()

""" #debugging
print(np.unique(y)) """


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#ravel if pandas dataframe is used
#y_train = y_train.values.ravel()
#y_test = y_test.values.ravel()

model = LogisticRegression(max_iter = 5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)





#create plots to check model quality

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = "d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
