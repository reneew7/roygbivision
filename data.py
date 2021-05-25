"""
Play with dataset
"""
import uwimg
import numpy as np
import pandas as pd

df = pd.read_csv('colors.csv')
df.columns = ['name', 'formatted_name', 'hex', 'r', 'g', 'b']
data = pd.get_dummies(df, columns=['hex'])

train_data = data.sample(frac=0.9, random_state=5)
test_data = data.drop(train_data.index)

columns = data.columns
labels = columns[5:len(columns)]
color_names = data[columns[0:2]]
train_rgb_data = train_data[columns[2:5]]
train_labels = train_data[labels]
test_rgb_data = test_data[columns[2:5]]
print(train_rgb_data)
print(train_labels)
