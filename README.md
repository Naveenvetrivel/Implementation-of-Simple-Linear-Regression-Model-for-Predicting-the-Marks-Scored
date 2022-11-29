# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1.Use the standard libraries in python.
2.Set variables for assigning dataset values. 
3.Import LinearRegression from the sklearn. 
4.Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of graph.
6.Compare the graphs and hence we obtain the LinearRegression for the given datas.
~~~
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEENKUMAR V
RegisterNumber:  212221230068

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset= pd.read_csv('student_scores.csv')
dataset.head()
X=dataset.iloc[:,:-1].values
X
y=dataset.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
y_test 
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hour vs scores(Training set)")
#plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='red')#plotting the regression line
plt.title("Hours vs scores(Testing set)")
#plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![image](https://user-images.githubusercontent.com/94165322/202891919-6d40519d-f27b-451f-9f7e-97a7acef7512.png)
![image](https://user-images.githubusercontent.com/94165322/202891925-f7c521eb-4876-477e-9a32-4ef5bdf39272.png)
![image](https://user-images.githubusercontent.com/94165322/202891931-57b06d9e-744f-4c88-ba89-fa606090ca48.png)
![image](https://user-images.githubusercontent.com/94165322/202891935-aa33ba0c-0b60-45e3-a033-d1683f06f727.png)
![image](https://user-images.githubusercontent.com/94165322/202891940-e11eb62f-fd92-4f2c-bafa-39156ed01bdd.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
