from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def scatter(x1, y1):
    plt.scatter(x1, y1)
    plt.grid(True)
    plt.xlabel(' X ')
    plt.ylabel(' Y ')
    plt.title('SCATTER PLOT')
    plt.show()

# Read Data
df = pd.read_csv('nonlinear_dataset.csv')

# PLOTS
scatter(df['x1'], df['y'])
scatter(df['x2'], df['y'])

# Normalize Data
x = df['x1'] / max(df['x1'])
y = df['y'] / max(df['y'])

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

def func(x1, a, b, c):
    return a + b * np.exp(c * x1)

params_x1, pcov_x1 = curve_fit(func, x_train, y_train)
a1, b1, c1 = params_x1

# Generate smooth curve for plotting
x_smooth = np.linspace(x.min(), x.max(), 100)
y_smooth = func(x_smooth, a1, b1, c1)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x.values, y.values, alpha=0.5, label='Data')
plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Fitted curve')
plt.xlabel('X1 (normalized)')
plt.ylabel('Y (normalized)')
plt.title('Nonlinear Regression Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


pred_y = func(x_test, a1, b1, c1)
r2 = r2_score(y_test, pred_y)
print(r2)
