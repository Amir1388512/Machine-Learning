import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


df = pd.read_csv('multiple_regression_dataset.csv')


plt.scatter(df['TV_Advertising'], df['Sales'])
plt.grid(True)
plt.show() # good

plt.scatter(df['Radio_Advertising'], df['Sales'])
plt.grid(True)
plt.show() # good

plt.scatter(df['SocialMedia_Advertising'], df['Sales'])
plt.grid(True)
plt.show() #Failed


plt.scatter(df['Price'], df['Sales'])
plt.grid(True)
plt.show() # good

np.random.seed(42)
msk = np.random.rand(len(df)) > 0.8
train = df[~msk]
test = df[msk]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(train['TV_Advertising'], train['Sales'])
ax.scatter(train['Radio_Advertising'], train['Sales'])
ax.scatter(train['Price'], train['Sales'])
plt.grid()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(test['TV_Advertising'], test['Sales'])
ax.scatter(test['Radio_Advertising'], test['Sales'])
ax.scatter(test['Price'], test['Sales'])
plt.grid()
plt.show()

x_train = np.asanyarray(train[['TV_Advertising', 'Radio_Advertising', 'Price']])
y_train = np.asanyarray(train[['Sales']])

x_test = np.asanyarray(test[['TV_Advertising', 'Radio_Advertising', 'Price']])
y_test = np.asanyarray(test[['Sales']])

ones = np.ones((len(train), 1) ,dtype=int)
x = np.hstack((ones, x_train))
get_coef = np.linalg.pinv(x.T @ x) @ (x.T @ y_train)
get_coef = get_coef.flatten()

li = [get_coef[0] + (get_coef[1] * x1) + (get_coef[2] * x2) + (get_coef[3] * x3) for x1,x2,x3 in zip(x_test[:, 0], x_test[:, 1], x_test[:, 2])]

MAE, MSE ,rse_sorat, rse_makhraj, rae_sorat, rae_makhraj = 0,0,0,0,0,0
y_test_mean = y_test.mean()

for pr , ans in zip(li, y_test):
    MAE += abs(pr - ans)
    MSE += (pr - ans) ** 2

    rae_sorat += abs(pr - ans)
    rae_makhraj += abs(ans - y_test_mean)

    rse_sorat += (pr - ans) ** 2
    rse_makhraj += (ans - y_test_mean) ** 2

print(f'MAE : {MAE / len(li)}')
print(f'MSE : {MSE / len(li)}')
print(f'RMSE : {np.sqrt(MSE / len(li))}')
RAE = rae_sorat / rae_makhraj
RSE = rse_sorat / rse_makhraj

print(f'RAE : {RAE}')
print(f'RSE : {RSE}')

r2_ = 1 - RSE
print(r2_)