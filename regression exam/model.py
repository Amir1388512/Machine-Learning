# Imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

# Read Dataset
data = pd.read_csv('1632300362534233.csv')

# Split Data
x = data.drop(['Price', 'Price(USD)'], axis=1)
y = data['Price(USD)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Split Columns
numeric_columns = x.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = x.select_dtypes(include=['object', 'bool']).columns


# Transformer
numeric_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocess
process = ColumnTransformer([
    ('num', numeric_transform , numeric_columns),
    ('cat', categorical_transform, categorical_columns)
])

# Model
model = Pipeline([
    ('preprocess', process),
    ('regressor', LinearRegression())
])

# Model Fit
model.fit(x_train, y_train)

# Prediction
predict = model.predict(x_test)

# Accuracy
print(f'MSE : {mean_squared_error(y_test, predict)}')
print(f'R2 : {r2_score(y_test, predict)}')