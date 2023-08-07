import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 

dataset = pd.read_csv("D:\Salary_Data.csv")
x = dataset.iloc[:,:-1].values       #input , 2D matrix , -1 means all of them except the last column
y = dataset.iloc[:,1].values        #output , vector 1D
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30) # Takes input , output and test size
reg = LinearRegression()
f = reg.fit(x_train,y_train)
b = reg.intercept_
A = reg.coef_ #Slope
y_pred = reg.predict(x_test)         #Regression is used to predict the output

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,reg.predict(x_train) , color='blue')
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_test,reg.predict(x_test) , color='blue')
plt.title("Salary VS Experience (Training set)")
plt.xlabel("Years of experince")
plt.ylabel("Salary")
plt.show()

mse = mean_squared_error(y_test,y_pred)


def mm(x,y):
    dif = 0
    for i in range (len(x)):
        dif=dif+(x[i] - y[i])**2

    return dif/len(x)

z = reg.predict([[8.8]]) #2D matrix

print("Salary = ",z)

accuracy = r2_score(y_test, y_pred)
print(f"R2 Score: {accuracy}")    

