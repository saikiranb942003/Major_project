import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
data = pd.read_csv('diabetes.csv')
data.head()
data.shape
data.isnull().sum()

data.corr()
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot = True)
X = data.iloc[:,:-1]
y = data['Outcome']
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
print("Train Set: ", X_train.shape, y_train.shape)
print("Test Set: ", X_test.shape, y_test.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, model.predict(X_test))*100)
import pickle
pickle.dump(model, open("diabetes.pkl",'wb'))