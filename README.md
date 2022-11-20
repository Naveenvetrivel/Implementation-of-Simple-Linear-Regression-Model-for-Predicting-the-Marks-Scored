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

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/Placement_Data.csv')
print(dataset.iloc[3])

print(dataset.iloc[0:4])

print(dataset.iloc[:,1:3])

#implement a simple regression model for predicting the marks scored by students
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')

#implement a simple regression model for predicting the marks scored by students
#assigning hours to X& Scores to Y
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title("Traning set (H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,reg.predict(X_test),color="pink")
plt.title("Test set (H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("MES = ",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
![image](https://user-images.githubusercontent.com/94165322/202891919-6d40519d-f27b-451f-9f7e-97a7acef7512.png)
![image](https://user-images.githubusercontent.com/94165322/202891925-f7c521eb-4876-477e-9a32-4ef5bdf39272.png)
![image](https://user-images.githubusercontent.com/94165322/202891931-57b06d9e-744f-4c88-ba89-fa606090ca48.png)
![image](https://user-images.githubusercontent.com/94165322/202891935-aa33ba0c-0b60-45e3-a033-d1683f06f727.png)
![image](https://user-images.githubusercontent.com/94165322/202891940-e11eb62f-fd92-4f2c-bafa-39156ed01bdd.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
