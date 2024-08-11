import numpy as np
import pandas as pd


df  = pd.read_csv('Crop_recommendation.csv')


x = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=14)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()


model.fit(x_train,y_train)


model_predictions = model.predict(x_train)
