import numpy as np
import random
import io
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


print(tf.__version__)


path = "infinite_jest_text.txt"

with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
    
text = text.replace("\n", " ")
print("Corpus length: {}".format(len(text)))

chars = sorted(list(set(text)))
print("Total characters: {}".format(len(chars)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []

# for every 40 characters create an example + its next char
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i+maxlen]) # this doesnt include i+mxlen
    next_chars.append(text[i + maxlen]) # this is i+mxlen
print("Number of sequences: {}".format(len(sentences)))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# for each character in each "sentence", encode it (as positive) in the 
# example. Set the label to be the next character after that "sentence"
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        layers.LSTM(64),
        layers.Dense(len(chars), activation="softmax")
    ])

optimiser = keras.optimizers.Adam()
model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])


def sample(preds, temperature=1.0):
    """Helper function to sample an index from a probability array."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) # normalise
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


epochs = 120
batch_size = 128

with open("output.txt", "w") as out:
    for epoch in range(epochs):
        model.fit(x, y, batch_size=batch_size, epochs=1)


        if epoch % 10 == 0:
            out.write("\n")
            out.write("Generating text after epoch: {0:d}\n".format(epoch))

            start_index = random.randint(0, len(text)-maxlen-1)
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                out.write("...Diversity: {}\n".format(diversity))

                generated = ""
                sentence = text[start_index : start_index+maxlen]
                out.write("...Generating with seed: '{}'\n".format(sentence))

                for i in range(400):
                    x_pred = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_indices[char]] = 1.0
                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = indices_char[next_index]
                    sentence = sentence[1:] + next_char
                    generated += next_char

                out.write("...Generated: {}\n\n".format(generated))


def generate_sequences(model, text, indices_char):   
    start_index = random.randint(0, len(text)-maxlen-1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("...Diversity: {}".format(diversity))
        
        generated = ""
        sentence = text[start_index : start_index+maxlen]
        print("...Generating with seed: '{}'".format(sentence))
        
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char
            
        print("...Generated: ".format(generated))
        print()
