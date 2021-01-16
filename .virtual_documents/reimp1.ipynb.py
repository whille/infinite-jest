import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


print("You are on TF{}.".format(tf.__version__))

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) == 0:
    print("You are not GPU accelerated.")
else:
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)


path = "infinite_jest_text.txt"

with open(path, "r") as f:
    text = f.read()
    
text = text.lower().replace("\n", " ")

unique_chars = sorted(list(set(text)))

idx_to_char = dict((i,c) for (i,c) in enumerate(unique_chars))
char_to_idx = dict((c, i) for (i, c) in enumerate(unique_chars))


maxlen = 40
stride = 3
sentences = []
next_chars = []

for i in range(0, len(text)-maxlen, stride):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])


print("Sentence: {}\nNext character: {}".format(sentences[25], next_chars[25]))


x = np.zeros((len(sentences), maxlen, len(unique_chars)))
y = np.zeros((len(sentences), len(unique_chars)))

# Let's now go through our sentences and characters and encode examples.
for sentence_index, sentence in enumerate(sentences):
    for char_index, char in enumerate(sentence):
        x[sentence_index, char_index, char_to_idx[char]] = 1
    y[sentence_index, char_to_idx[next_chars[sentence_index]]] = 1


print("One input char: {}".format(x[0][0]))
print("One output char: {}".format(y[0]))


which_sentence = 0
chars = []
for char_vector in x[which_sentence]:
    chars.append(unique_chars[np.argmax(char_vector)])
    
print("Input sentence: {}".format("".join(chars)))
print("Next char: {}".format(unique_chars[np.argmax(y[which_sentence])]))


print(x.shape)


batch_size = 128

# Creating the model is the simplest part of this notebook.
model = keras.Sequential(
[
    # FIXME: what's the dimension of this input supposed to be?
    keras.layers.Input((x.shape[0], x.shape[1]), batch_size, name="Input"), 
    keras.layers.LSTM(128, name="LSTM"),
    keras.layers.Dense(len(unique_chars), activation="softmax", name="Dense")
])


optimizer = keras.optimizers.Adam()

# what should the loss be? what is each loss good for?
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


model.fit(x, y, epochs=1)



