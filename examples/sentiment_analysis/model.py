import tensorflow as tf


# encoder vocab size is 8185
rnn_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(8185, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

RNNModel = lambda: rnn_model
