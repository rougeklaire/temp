import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("/home/yt0rk/Desktop/Uni/TUM/MIUG/Project/Dataset/NC_policing.csv")

#missing value handling and choosing features
data['drugs_related_stop'] = data['drugs_related_stop'].fillna('unknown') #too much data would be lost if rows just dropped
data.dropna(subset=['driver_age'], inplace=True) #enough data left if missing value rows ignored


#state is same for all, driver_race_raw is a duplicate, date&district nonsenitive (not our focus here)
#officer is kinda weird since do many values numerical ones that shouldn't have a weight
#search basis is one step before the actual search that influences the outcome

X_raw = data.drop(columns=['stop_outcome','state', 'stop_date', 'driver_race_raw', 'district', 'officer_id', 'search_basis'])
y_raw = data['stop_outcome']

#encoding
lencoder = LabelEncoder()
y_encoded = lencoder.fit_transform(y_raw)

gender_enc = lencoder.fit_transform(X_raw['driver_gender'])

#different data formats, use replacements
X_drugs = X_raw[['drugs_related_stop']]
replacements = {'unknown': 0, True :1}
X_drugs = X_drugs.replace(replacements)

multicat_features = ['driver_race', 'violation', 'search_type'] #redefine without binary

X_multi = X_raw[multicat_features] 
X_multi = pd.get_dummies(X_multi)

#combine into encoded df
X_encoded = X_raw.copy()
X_encoded.driver_gender = gender_enc
X_encoded.drugs_related_stop = X_drugs

X_encoded = X_encoded.reset_index(drop=True)  # Reset index to start from 0
X_multi = X_multi.reset_index(drop=True)      # Reset index to start from 0
X_encoded = pd.concat([X_encoded, X_multi], axis=1) 
X_encoded = X_encoded.drop(columns= multicat_features)

#scale 
scaler = StandardScaler()
X_encoded[['driver_age']] = scaler.fit_transform(X_encoded[['driver_age']])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

#specify model & train
model = LogisticRegression(max_iter=50000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#accuracy = accuracy_score(y_test, y_pred)

#y_frequency = y.value_counts()
#print(y_frequency)

# Create plots to check model quality

#print(f"Accuracy Score: {accuracy}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(conf_matrix, annot=True, fmt="d")
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.show()

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f'Accuracy: {accuracy}')

# Precision
precision = TP / (TP + FP)
print(f'Precision: {precision}')

# Recall or Sensitivity
recall = TP / (TP + FN)
print(f'Recall: {recall}')

# Specificity
specificity = TN / (TN + FP)
print(f'Specificity: {specificity}')

# F1-Score
f1 = 2 * (precision * recall) / (precision + recall)
print(f'F1-Score: {f1}')
