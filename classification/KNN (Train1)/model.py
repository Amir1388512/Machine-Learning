# Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv('knn_dataset.csv')

x = data.drop('Class', axis=1)
y = data['Class']

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=42)

k = range(1,9)
score = []
for i in k:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_x,train_y)
    score.append(f1_score(test_y, model.predict(test_x), average="macro"))
print(score)