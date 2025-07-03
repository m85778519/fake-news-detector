import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle

# 1. Load data
df = pd.read_csv("fake_or_real_news.csv")

# 2. Clean + prep
texts = df['text'].astype(str).values
labels = df['label'].values

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=500)

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2)

# 4. Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 128, input_length=500),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)

# 5. Save model + files
model.save("model.h5")

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("label_encoder.pickle", "wb") as file:
    pickle.dump(label_encoder, file)
