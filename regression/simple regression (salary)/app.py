from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv('final_exam_regression_dataset.csv')
print(data.corr())  # Experience_Years is the best

# But Let's check them on the chart
plt.scatter(data['Experience_Years'],data['Salary'])
plt.ylabel('Salary')
plt.xlabel('Experience Years')
plt.grid(True)
plt.show()
# -------------------------------------------------
plt.scatter(data['Education_Years'],data['Salary'])
plt.ylabel('Salary')
plt.xlabel('Education Years')
plt.grid(True)
plt.show()
# -------------------------------------------------
plt.scatter(data['Projects_Completed'],data['Salary'])
plt.ylabel('Salary')
plt.xlabel('Projects Completed')
plt.grid(True)
plt.show()
# -------------------------------------------------
plt.scatter(data['Certifications'],data['Salary'])
plt.ylabel('Salary')
plt.xlabel('Certifications')
plt.grid(True)
plt.show()
# -------------------------------------------------
# divided data into train and multiple regression (Salary)
np.random.seed(42)
msk = np.random.rand(len(data)) > 0.8
test = data[msk]
train = data[~msk]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(train['Experience_Years'], train['Salary'], color='b')
ax.scatter(test['Experience_Years'], test['Salary'], color='r')
plt.grid(True)
plt.show()

x_train = np.asanyarray(train[['Experience_Years']])
y_train = np.asanyarray(train[['Salary']])

x_test = np.asanyarray(test[['Experience_Years']])
y_test = np.asanyarray(test[['Salary']])

# create model automation with scikit learn
model = LinearRegression()
model.fit(x_train,y_train)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(train['Experience_Years'], train['Salary'], color='b')
ax.scatter(test['Experience_Years'], test['Salary'], color='r')
ax.plot(x_train, model.coef_[0][0] * x_train + model.intercept_[0])
plt.grid(True)
plt.show()

prediction = model.predict(x_test)
r2 = r2_score(y_test,prediction)

print(r2)


# Create manualy

x_train = np.asanyarray(train['Experience_Years'])
y_train = np.asanyarray(train['Salary'])

x_test = np.asanyarray(test['Experience_Years'])
y_test = np.asanyarray(test['Salary'])

up , down = 0,0
x_train_mean = x_train.mean()
y_train_mean = y_train.mean()
for x, y in zip(x_train,y_train):
    up += (x - x_train_mean) * (y - y_train_mean)
    down += (x - x_train_mean) ** 2

theta1 = up / down
theta0 = y_train_mean - (x_train_mean * theta1)

print(f'formula : x{theta1} + {theta0}') # my line formula

li = [theta0 + (theta1 * i) for i in x_test]

y_test_mean = y_test.mean()


MAE, MSE ,rse_sorat, rse_makhraj, rae_sorat, rae_makhraj = 0,0,0,0,0,0


for pr , ans in zip(li, y_test):
    MAE += abs(pr - ans)
    MSE += (pr - ans) ** 2

    rae_sorat += abs(pr - ans)
    rae_makhraj += abs(ans - y_test_mean)

    rse_sorat += (pr - ans) ** 2
    rse_makhraj += (ans - y_test_mean) ** 2

print(f'MAE : {MAE / len(li)}')
print(f'MSE : {MSE / len(li)}')
print(f'RMSE : {math.sqrt(MSE / len(li))}')
RAE = rae_sorat / rae_makhraj
RSE = rse_sorat / rse_makhraj

print(f'RAE : {RAE}')
print(f'RSE : {RSE}')

r2_ = 1 - RSE
print(r2_)