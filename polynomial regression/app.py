# Imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
