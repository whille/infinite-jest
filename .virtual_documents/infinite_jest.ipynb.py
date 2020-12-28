import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


get_ipython().run_line_magic("load_ext", " tensorboard")


path_to_file = "infinite_jest_text.txt"
with open(path_to_file, "r") as text_in:
    text = text_in.read()
    
vocab = sorted(set(text)) # get unique words, sort alphabetically


# for each character in the text, associate an integer value
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# represent the text as an array of integers
text_as_int = np.array([char2idx[c] for c in text])


seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# create a Tensorflow Dataset type from the text-as-array-of-ints
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


# creates batches from this dataset, drops the last batch if too small
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    """Takes a string 'abcde' and returns 'abcd', 'bcde'"""
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# our dataset will have batches of these types of split
# strings, because we want to predict the next character
dataset = sequences.map(split_input_target)


BATCH_SIZE = 64
BUFFER_SIZE = 10000

# shuffle the dataset batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


print(dataset)


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 512


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]), 
        keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        keras.layers.Dense(vocab_size)
    ])
    
    return model


model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)


model.summary()


def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(loss=loss,
             optimizer="adam", 
             metrics=["accuracy"])


EPOCHS = 10
log_dir = "logs/fit/" + datetime.datetime.now().strftime("get_ipython().run_line_magic("Y%m%d-%H%M%S")", "")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(dataset, 
                   epochs=EPOCHS,
                   callbacks=[tensorboard_callback])



