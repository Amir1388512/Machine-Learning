import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('house_prices_regression.csv')

# 80/20 split
msk = np.random.rand(len(data)) > 0.8
train = data[~msk]
test = data[msk]

x_train = np.asanyarray(train['Size_sqm'])
y_train = np.asanyarray(train['Price'])

x_test = np.asanyarray(test['Size_sqm'])
y_test = np.asanyarray(test['Price'])



s,kh = 0,0
x_mean =  x_train.mean()
y_mean = y_train.mean()

for x,y in zip(x_train,y_train):
    s += (x - x_mean) * (y - y_mean)
    kh += (x - x_mean) ** 2

theta1 = s / kh
theta0 = y_mean - (theta1 * x_mean)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x_train, y_train, color='red', label='Training Data')
ax1.scatter(x_test, y_test, color='blue', label='Test Data')
ax1.plot(x_train ,  theta0 + theta1 * x_train, 'g--')
ax1.set_xlabel('Size (sqm)')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True)
plt.show()

li = [theta1 * i + theta0 for i in x_test]

sorat, makhraj = 0,0
for predicted, answer  in zip(li,y_test):
    sorat += (answer - predicted) ** 2
    makhraj += (answer - y_mean) ** 2

print(1 - sorat / makhraj)
