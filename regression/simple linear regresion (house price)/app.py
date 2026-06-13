
# Imports Libraries needed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('house_prices_regression.csv')

# make some chart to see my datas
# after some check we understood that the size is the best feature to create our model
plt.scatter(data['Size_sqm'], data['Price'], color='blue')
plt.xlabel('size')
plt.ylabel('price')
plt.grid()
plt.show()

msk = np.random.rand(len(data)) > 0.8
test = data[msk]
train = data[~msk]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(train.Size_sqm, train.Price, color='red')
ax1.scatter(test.Size_sqm, test.Price, color='blue')
plt.show()

x_train = np.asanyarray(train[['Size_sqm']])
y_train = np.asanyarray(train[['Price']])

model = LinearRegression()
model.fit(x_train, y_train)

x_test = np.asanyarray(test[['Size_sqm']])
y_test = np.asanyarray(test[['Price']])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(train.Size_sqm, train.Price, color='red')
ax1.scatter(test.Size_sqm, test.Price, color='blue')
ax1.plot(x_train, model.intercept_[0] + model.coef_[0][0] * x_train, 'g--')
plt.show()

per = model.predict(x_test)
accu = r2_score(y_test, per)

print(accu)
print(f"R2 Score = {accu:.3f}")
