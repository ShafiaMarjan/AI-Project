# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('resdata.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print(dataset.head())

print("Length of the dataset: ",len(dataset))


#Fill the missing val with imputer.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:,:])
X[:,:]=imputer.transform(X[:,:])


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#simple regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test result
y_pred = regressor.predict(X_test)

#plot graphical result train set
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('GPA to CGPA')
plt.xlabel('GPA')
plt.ylabel('CGPA')
plt.show()

#plot graphical result test set
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('GPA to CGPA')
plt.xlabel('GPA')
plt.ylabel('CGPA')
plt.show()

for i in range(len(y_test)):
    print("Actual ",i,":",y_test[i],"\tPredicted ",i,":",y_pred[i])
