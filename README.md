# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score


## Program:
```



import pandas as pd
 data=pd.read_csv("Employee.csv")
 print("data.head():")
 data.head()
 print("data.info():")
 data.info()
 print("data.info():")
 data.info()
 print("data value countrs():")
 data["left"].value_counts()
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 print("data.head() for salary:")
 data["salary"]=le.fit_transform(data["salary"])
 data.head()
 print("x.head():")
 x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",]]
 x.head()
 y=data["left"]
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
 from sklearn.tree import DecisionTreeClassifier
 dt=DecisionTreeClassifier(criterion="entropy")
 dt.fit(x_train,y_train)
 y_pred=dt.predict(x_test)
 print("Accuracy value:")
 from sklearn import metrics
 accuracy=metrics.accuracy_score(y_test,y_pred)
 accuracy
 print("Data prediction:")
 dt.predict([[0.5,260,0,2]])
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sudhindra.R
RegisterNumber: 24901168
*/
```

## Output:

![Screenshot 2024-11-29 032457](https://github.com/user-attachments/assets/26135ff7-e48c-4c9c-a4bb-aac2f5d2a709)
![Screenshot 2024-11-29 032510](https://github.com/user-attachments/assets/10d0f168-2346-4edc-ba10-83cb7c1b637f)
![Screenshot 2024-11-29 032530](https://github.com/user-attachments/assets/deb247d6-3c5d-4789-90e7-a02b0660d956)
![Screenshot 2024-11-29 032542](https://github.com/user-attachments/assets/02613c28-aa9a-47b6-b54e-98469cea6a7e)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
