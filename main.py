import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

data = keras.datasets.imdb

(train_dt, train_lb), (test_dt, test_lb) = data.load_data(num_words=88_000)  # taking 10_000 most repeatable words

word_index = data.get_word_index()  # key-value dictionary (key: word, value: number)

# Increasing every value by 3 in a word index e.g. 'the' was 1 and then 'the' will be 4
# Swapping key-values together e.g. word_index['plot'] = 114 but I want reversed_word_index[114] = plot
word_index = {k: (v+3) for k, v in word_index.items()}

# Reserving values 0-3 to give them another meaning. Words denoted previously by these values won't be changed,
# only shifted to the right
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3


# Reversing key and values together in dictionary
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])


# Preprocessing data: defined padding (making it the same length)
train_dt = keras.preprocessing.sequence.pad_sequences(train_dt, maxlen=2494, value=word_index['<PAD>'], padding='post')
test_dt = keras.preprocessing.sequence.pad_sequences(test_dt, maxlen=2494, value=word_index['<PAD>'], padding='post')


def decoder(text_item, index_of_words):
    return " ".join([index_of_words[x] for x in text_item])


# defining model
model = keras.Sequential()
model.add(keras.layers.Embedding(88_000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()  # prints string summary of network

# binary_crossentropy because of sigmoid activation function which returns probability
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data validation
x_validation = train_dt[:10_000]
x_train = train_dt[10_000:]
y_validation = train_lb[:10_000]
y_train = train_lb[10_000:]

model.fit(x_train, y_train, batch_size=512, epochs=45, verbose=1, validation_data=(x_validation, y_validation))

results = model.evaluate(test_dt, test_lb)


