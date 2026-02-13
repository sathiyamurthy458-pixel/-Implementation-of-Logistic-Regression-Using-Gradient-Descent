# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

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
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sathiya Murthy k
RegisterNumber:  212225100047
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts ()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le. fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
"time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x. head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt. fit(x_train, y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics. accuracy_score(y_test, y_pred)
accuracy

0.9843333333333333

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

<img width="1501" height="539" alt="image" src="https://github.com/user-attachments/assets/b4256317-b3d5-47ae-ad5f-10b320839720" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

