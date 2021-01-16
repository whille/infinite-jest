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

for i in range(len(text)-maxlen):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])


print("Sentence: {}\nNext character: {}".format(sentences[25], next_chars[25]))


    
x = np.zeros(((len(sentences), maxlen, len(unique_chars))))

y = np.zeros((len(sentences), len(unique_chars)))

shape_of_examples = None # placeholder—I need to find out what my inputs look like


batch_size = 128

# Creating the model is the simplest part of this notebook.
model = keras.Sequential(
[
    # FIXME: what's the dimension of this input supposed to be?
    keras.layers.Input(shape_of_examples, batch_size), 
    keras.layers.LSTM(128),
    keras.layers.Dense(len(unique_chars), activation="softmax")
])


optimizer = keras.optimizers.Adam()

# what should the loss be? what is each loss good for?
model.compile(loss="")



