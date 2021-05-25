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


def min_color(r, g, b):
    error = np.infty
    track_rgb = [r, g, b]
    idx = 0
    for c in range(df.shape[0]):
        red = df.loc[c, 'r']
        gr = df.loc[c, 'g']
        blue = df.loc[c, 'b']
        r_e = (red - r) ** 2
        g_e = (gr - g) ** 2
        b_e = (blue - b) ** 2
        total_e = r_e + g_e + b_e
        if total_e < error:
            idx = c
            error = total_e
            track_rgb = [red, gr, blue]
    print(track_rgb, error)
    print(df.loc[idx, 'name'])


min_color(200, 1, 1)

