## Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Import dataset
dataset = pd.read_csv("Position_Salaries.csv")

# matrix of features
X = dataset.iloc[:, 1:2].values # here, column index for dataset was needed, but added to make the DATATYPE AS MATRIX
# Dependent variable vector
y = dataset.iloc[:, 2].values

# Plotting level and salary, to DECIDE MODEL TO USE
plt.scatter(X, y, color = "red")
plt.show()

# polynomial regression degree 4
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(degree = 4) # to make X^0, X^1, X^2
X_poly_degree4 = poly_feat.fit_transform(X)

linearregressor_4 = LinearRegression()
linearregressor_4 = linearregressor_4.fit(X_poly_degree4, y)
resultsalary_lin_reg_4 = linearregressor_4.predict(np.array([1, 6.5, 6.5 ** 2, 6.5**3, 6.5**4]).reshape(1, -1))

# visualising  polynomial regression degree 4
plt.scatter(X, y, color = "red")
plt.scatter(6.5, resultsalary_lin_reg_4, color="green") #ACTUAL RESULT FROM polynomial REGRESSION
plt.plot(X, linearregressor_4.predict(X_poly_degree4), color='blue')
plt.title('Level Vs Salary (POLYNOMIAL, degree 3, VERY ACCURATE FIT)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

