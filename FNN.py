import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

WineDF = pd.read_csv('./red-wine.csv')
df_train = WineDF.sample(frac=0.8,random_state=0)
df_test = WineDF.drop(df_train.index)


# normalize the data set

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_test = (df_test - min_) / (max_ - min_)

x_train = df_train.drop('quality',axis=1)
x_test = df_test.drop('quality',axis=1)

y_train = df_train['quality']
y_test = df_test['quality']

# now build the Neural Nets.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(11, activation='relu',input_shape=[11]),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(1)
])

early_stopping = tf.keras.callbacks.EarlyStopping(
    min_delta= 0.001,
    patience=20,
    restore_best_weights=True
)

model.compile(
    optimizer='adam',
    loss='mae',
)

History = model.fit(x_train,y_train,
          batch_size=100,
          epochs=600,
            validation_data=(x_test,y_test),
            callbacks=[early_stopping],
          )

model_df = pd.DataFrame(History.history)

model_df.loc[: , ['loss','val_loss']].plot()

plt.show()