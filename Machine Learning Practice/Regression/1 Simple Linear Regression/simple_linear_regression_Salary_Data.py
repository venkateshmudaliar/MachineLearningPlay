# Simple Linear Regression
# y = a0 + a1.x
# response = coeff1 + coeff2.Feature
# Goal - Find a0 and a1 which best fits the model
# Coeff: minimizing "sum of sqaured errors(residuals)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
print("Importing the dataset \n")
dataset = pd.read_csv('Data/Salary_Data.csv')


# Analysing Dataset
print("Analysing Dataset \n")
print("Dataset Shape: ",dataset.shape)
print("Dataset Describe: ",dataset.describe())

print(" \n")


print("Plot Dataset:")
dataset.plot(x='YearsExperience', y='Salary', style='o')
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

sns.pairplot(dataset, x_vars='YearsExperience', y_vars='Salary')
plt.show()



# Preparing Dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(" \nSplitting the dataset into the Training set and Test set \n")
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


print("Training the Simple Linear Regression model on the Training set \n")
# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# print the coefficients
# a0 - is the intercept (the value of  y  when  x =0)
# a1 - is the slope (the change in  y  divided by change in  x )
# linear regression model basically finds the best value for the intercept and slope
print(" \nThe value of Coefficients a0 and a1:  \n",regressor.intercept_,regressor.coef_ )


print("Predicting the Test set results \n")
# Predicting the Test set results
y_pred = regressor.predict(X_test)


print("Visualising the Training set results \n")
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print("\nVisualising the Test set results \n")
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


score_value_train = regressor.score(X_train, y_train)
score_value_test = regressor.score(X_test, y_test)

print(" \nScore Values")
print("Training Score: ",score_value_train)
print("Testing Score: ",score_value_test)


print("\nDisplaying Test and Prediction Values:")
#for d, c in zip(y_test, y_pred):
#    print (d ,"\t", c)



#res = "\n".join("{} \t {} \t {}".format(x, y, z) for x, y, z in zip(y_test, y_pred, y_test-y_pred))
#print(res)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
print("\n\n","-------------")

print("\n")
experience_val = float(input("Enter the Experience (years):"))


# The value of Coefficients a0 and a1:  26816.19224403119 [9345.94244312]
# Manual Prediction:
manual_predicted_salary = 26816.19224403119 + 9345.94244312*experience_val
print("Predicated Salary for Experience (Manually):",experience_val," = ",manual_predicted_salary )


# Regressor Prediction: 
predicted_salary = regressor.predict([[experience_val]])
print("Predicated Salary for Experience (Regressor):",experience_val," = ",predicted_salary )


from sklearn import metrics
print("\nMean Absolute Error: ",metrics.mean_absolute_error(y_test, y_pred))
print("\nMean Squared Error: ",metrics.mean_absolute_error(y_test, y_pred))
print("\nRoot Mean Squared Error: ",np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))

# Accuracy
accuracy = regressor.score(X_test,y_test)
print("\nAccuracy: ",accuracy*100,'%')