import numpy as np
import tensorflow as tf
from tensorflow import keras

data = keras.datasets.imdb

(train_dt, train_lb), (test_dt, test_lb) = data.load_data(num_words=10_000)  # taking only 10_000 reviews from dataset

word_index = data.get_word_index()  # key-value (key: word, value: number) in a form of tuple
# Breaking up tuple to make it dictionary (that's why I put it in {} and said word_index.items()),
# adding + 3 to every int value
word_index = {k: (v+3) for k, v in word_index.items()}  # word_index['plot'] = 114 but I want word_index[114] = plot
# Values in that dictionary start at 0, but I want to give 0,1,2,3 values another meaning, which is described below
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
    # Swapping number with a corresponding word to it
    len_arr = []
    for item in range(len(text_item)):
        string = ''
        str_length = 0
        for word in text_item[item]:
            string = string + index_of_words[word] + ' '
            str_length += 1
        # print(string)
        len_arr.append(str_length)
    # print('Number of most words in review: ', max(len_arr))


decoder(train_dt, reversed_word_index)

