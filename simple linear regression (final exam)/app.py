import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('student_scores_regression_1000.csv')
print(data.corr(numeric_only=True))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(data['Study_Hours'], data['Final_Score'], color='b')
# ax1.scatter(data['Assignments'], data['Final_Score'], color='r')
plt.grid(True)
plt.ylabel('Final Score')
plt.show()

msk = np.random.rand(len(data)) > 0.8
test = data[msk]
train = data[~msk]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(test['Study_Hours'], test['Final_Score'], color='b')
ax1.scatter(train['Study_Hours'], train['Final_Score'], color='r')
plt.grid(True)
plt.ylabel('Final Score')
plt.show()

x_train = np.asanyarray(train['Study_Hours'])
y_train = np.asanyarray(train['Final_Score'])

x_test = np.asanyarray(test['Study_Hours'])
y_test = np.asanyarray(test['Final_Score'])


x_train_mean = x_train.mean()
y_train_mean = y_train.mean()

up , down = 0,0

for x,y in zip(x_train,y_train):
    up += (x - x_train_mean) * (y - y_train_mean)
    down += (x - x_train_mean) ** 2

theta1 = up / down
theta0 = y_train_mean - (theta1 * x_train_mean)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(train['Study_Hours'], train['Final_Score'], color='r')
ax1.plot(x_train, theta0 + (theta1 * x_train))
plt.grid(True)
plt.ylabel('Final Score')
plt.show()

li = [theta0 + (theta1 * i) for i in x_test]

sorat, makhraj = 0,0
y_test_mean = y_test.mean()
for pred, ans in zip(li, y_test):
    sorat += (pred - ans) ** 2
    makhraj += (ans - y_test_mean) **2

RSE = sorat / makhraj
r2 = 1 - RSE

print(r2)