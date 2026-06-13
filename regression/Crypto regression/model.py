# Imports
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd



# Read Dataset
data = pd.read_csv('crypto_regression_dataset.csv')


# Spilt Data
x = data.drop(['PriceUSD', 'AgeDays', 'DevActivity'], axis=1)
y = data['PriceUSD']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)

numeric_cols = x.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = x.select_dtypes(include='object').columns

num_transform = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transform = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

process = ColumnTransformer([
    ('num', num_transform, numeric_cols),
    ('cat', categorical_transform, categorical_cols)
])


model = Pipeline([
    ('preprocess', process),
    ('regressor', LinearRegression())
])
model.fit(x_train,y_train)


predict = model.predict(x_test)
print(r2_score(y_test, predict))


