import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
# https://medium.com/analytics-vidhya/building-rgb-color-classifier-part-1-af58e3bcfef7

df = pd.read_csv('rgb_color_labels.csv')
data = pd.get_dummies(df, columns=['label'])
train_data = data.sample(frac=0.9, random_state=5)
test_data = data.drop(train_data.index)

columns = data.columns
labels = columns[3:len(columns)]

train_rgb_data = train_data[columns[0:3]]
train_labels = train_data[labels]
test_rgb_data = test_data[columns[0:3]]
test_labels = test_data[labels]
# print(train_rgb_data)
# print(train_labels)

model = tf.keras.Sequential([
    layers.Dense(3, kernel_regularizer=regularizers.l2(0.001), activation='relu',
                 input_shape=[len(train_rgb_data.keys())]),
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(11)
])
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

m = model.fit(x=train_rgb_data, y=train_labels, validation_split=0.2, epochs=50, batch_size=2048)



