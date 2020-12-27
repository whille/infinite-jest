import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


with open("infinite_jest_text.txt", 'r') as file_in:
    ij_text = file_in.readlines()
    
print(len(ij_text))
print(ij_text[250:260])


# parameters for the tokeniser and embedding
vocab_size = 10000
embedding_dimensions = 64
max_length = 128
trunc_type = "post"
oov_token = "<OOV>"


tokeniser = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokeniser.fit_on_texts(ij_text)

word_index = tokeniser.word_index

sequences = tokeniser.texts_to_sequences(ij_text)
padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)


print(len(word_index))
print(sequences[0])



