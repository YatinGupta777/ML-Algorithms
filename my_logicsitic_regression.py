import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset

#x are independent variables that is the first 3
#y is dependent variable that is purchased

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,[2,3]].values
#[:,:-1] left means taking all the rows and right means taking all columns -1
y = dataset.iloc[:,4].values

#splitting dataset into traning set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Fitting Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

#predicitng the test results
y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


#Visualising the traning set results



