import numpy as np
import tensorflow as tf
from tensorflow import keras

data = keras.datasets.imdb

# words are represented by numbers
(train_dt, train_lb), (test_dt, test_lb) = data.load_data(num_words=88_000)  # taking 88_000 most repeatable words

word_index = data.get_word_index()  # key-value dictionary (key: word, value: number)

# Increasing every value by 3 in a word index e.g. 'the' was 1 and then 'the' will be 4
word_index = {k: (v+3) for k, v in word_index.items()}

# Reserving values 0-3 to give them another meaning. Words denoted previously by these values won't be changed,
# only shifted to the right
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3


# Reversing key and values together in dictionary
# Swapping key-values together e.g. word_index['plot'] = 114 but I want reversed_word_index[114] = plot
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Preprocessing data: defined padding (making it the same length)
train_dt = keras.preprocessing.sequence.pad_sequences(train_dt, maxlen=2494, value=word_index['<PAD>'], padding='post')
test_dt = keras.preprocessing.sequence.pad_sequences(test_dt, maxlen=2494, value=word_index['<PAD>'], padding='post')


def decoder(text_item, index_of_words):
    return " ".join([index_of_words[x] for x in text_item])


'''
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
'''
model = keras.models.load_model('IMDBmodel.h5')


def external_review(path):
    with open(path) as f:
        i = 0
        for line in f.readlines():
            num_words = len(line.split())
            print('Total number of words in file: ', num_words)
            # Note: size of the array can't be greater than the one declared in model before (2494)
            arr = np.ones(shape=num_words, dtype='int32')
            for word in line.lower().split():
                removed_char = word.replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace(':', '')
                number = word_index.get(removed_char, 2)
                arr[i] = number
                i += 1
            # first parameter is a list - "predict" will take all of the words together rather than making prediction on
            # a singular word (whether is positive or negative)
            arr = keras.preprocessing.sequence.pad_sequences([arr], maxlen=2494, value=word_index['<PAD>'], padding='post')
            np.set_printoptions(threshold=np.inf)
            predict = model.predict(arr)
            print('Probability review is positive: ', predict[0])


external_review('user_review.txt')


'''
test_review = test_dt[5]
print('Decoded review: ')
print(decoder(test_review, reversed_word_index))
predict = model.predict([test_review])
print('Predicted: ', str(predict[5]))
print('Actual: ', str(test_lb[5]))
print('Evaluation values: ', results)
'''