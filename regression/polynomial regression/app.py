# Imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('house_price_polynomial_dataset.csv')


plt.scatter(data['size'], data['price'])
plt.grid()
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

# i know that i can use train test split
np.random.seed(42)
msk = np.random.rand(len(data)) > 0.8
train = data[~msk]
test = data[msk]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(train['size'], train['price'])
ax.scatter(test['size'], test['price'])
plt.grid()
plt.show()


x_train = np.asanyarray(train[['size']])
y_train = np.asanyarray(train[['price']])

x_test = np.asanyarray(test[['size']])
y_test = np.asanyarray(test[['price']])

degree = [1,2,3,4,5]
r2 = []

for i in degree:
    poly = PolynomialFeatures(i)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    prediction = model.predict(x_test_poly)
    accuracy =  r2_score(y_test, prediction)
    r2.append(accuracy)
best_performance = {'Deg': degree[r2.index(max(r2))], 'r2' : max(r2)}


plt.figure()
plt.plot(degree, r2, marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("R² Score")
plt.title("Degree vs R²")
plt.grid(True)
plt.show()


