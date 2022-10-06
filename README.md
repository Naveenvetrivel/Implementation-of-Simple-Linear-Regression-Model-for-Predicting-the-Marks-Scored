# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEENKUMAR V
RegisterNumber:  212221230068
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('/content/std_score.csv')

dataset.head()

dataset.tail()

x=dataset.iloc[:,:-1].values   #.iloc[:,start_col,end_col]
y=dataset.iloc[:,1].values
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)

#for train values
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours Vs Score(Training set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

#for test values
y_pred=regressor.predict(x_test)
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,regressor.predict(x_test),color='black')
plt.title("Hours Vs Score(Test set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
```

## Output:
## TRAINING DATA
![image](https://user-images.githubusercontent.com/94165322/194205488-918bb0fa-675f-4898-99bc-9f2def1fe1d1.png)

## TESTING DATA
![image](https://user-images.githubusercontent.com/94165322/194205581-a130b3e4-1e2a-4a5f-a912-16994e9fe4a8.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
