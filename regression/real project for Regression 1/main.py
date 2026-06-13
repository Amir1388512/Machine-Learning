# Imports
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import pandas as pd

# Read Dataset
df = pd.read_csv('noisy_job_market.csv')

# Choice My Data
x = df.drop(['age', 'salary'], axis=1)
y = df['salary']

# Train / Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# split cols - FIXED: use x instead of df
numeric_col = x.select_dtypes(include=['int64', 'float64']).columns
categorical_col = x.select_dtypes(include='object').columns

# Operation
numeric_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Added handle_unknown
])

# Process
process = ColumnTransformer([
    ('num', numeric_transform, numeric_col),
    ('cat', categorical_transform, categorical_col)
])

# Model
model = Pipeline([
    ('preprocessor', process),
    ('regression', LinearRegression())
])

model.fit(x_train, y_train)
prediction = model.predict(x_test)

r2 = r2_score(y_test, prediction)
print(f"R² Score: {r2}")