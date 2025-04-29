# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries and load the dataset.

2.Define X and Y array.

3.Define a function for cost Function,cost and gradient.

4.Define a function to plot the decision boundary.

5.Define a function to predict the Regression value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Keerthana P
RegisterNumber:212223240069
*/
```

```
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/Placement_Data.csv')
dataset
```
### Output:
![image](https://github.com/user-attachments/assets/8ad3945b-5b72-462b-9766-c1e0f0d8f562)


```
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset
```
### Output:
![image](https://github.com/user-attachments/assets/2388ebd3-d3aa-4f48-843d-75c10a07b470)

```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
### Output:
![image](https://github.com/user-attachments/assets/070ea01b-9317-4bf7-bcda-dea5b95ec437)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
### Output:
![image](https://github.com/user-attachments/assets/73d574bc-b61f-4ce7-b943-f3519c04ac37)


```
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
```
### Output:
![image](https://github.com/user-attachments/assets/7c3ad37f-6414-4f2f-aa67-6a3700badc45)


```
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,Y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01, num_iterations=1000)
```
```
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
```
```
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
### Output:
![image](https://github.com/user-attachments/assets/4c4a8fda-759c-4675-8ea4-2f15492abe0f)

```
print(y_pred)
```
### Output:
![image](https://github.com/user-attachments/assets/5a463c48-6bc0-4a7e-8ddb-4fef319e3731)

```
print(Y)
```
### Output:
![image](https://github.com/user-attachments/assets/5124b6f0-0914-4d08-8e42-8b6422a7388f)

```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
### Output:
![image](https://github.com/user-attachments/assets/b0d60096-41bf-458d-95a6-0783c6efd19f)

```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
### Output:
![image](https://github.com/user-attachments/assets/54f9caf1-d2be-4e4b-8674-3c4dc748d945)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

