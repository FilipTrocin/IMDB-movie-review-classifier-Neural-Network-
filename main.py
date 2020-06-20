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

