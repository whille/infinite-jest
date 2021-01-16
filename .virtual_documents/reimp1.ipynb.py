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


# Creating the model is the simplest part of this notebook.
model = keras.Sequential(
[
    keras.layers.Input((maxlen, len(unique_chars)), name="Input"), 
    keras.layers.LSTM(128, name="LSTM"),
    keras.layers.Dense(len(unique_chars), activation="softmax", name="Dense")
])


optimizer = keras.optimizers.Adam()

# what should the loss be? what is each loss good for?
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


batch_size = 128

model.fit(x, y, epochs=1, batch_size=batch_size)


# Create a seed sentence.
seed_index = np.random.randint(len(text)-maxlen)
print("Seed index: {}".format(seed_index))

seed_sentence = text[seed_index:seed_index + maxlen]
ss_copy = text[seed_index:seed_index + maxlen]
print("Seed sentence: {}".format(seed_sentence))

for i in range(400):
    # Now to encode this sentence.
    pred_x = np.zeros((1, maxlen, len(unique_chars)), dtype=np.float32)
    for char_index, char in enumerate(seed_sentence):
            pred_x[0, char_index, char_to_idx[char]] = 1

    # Predict the next character, then add it to the sentence.
    preds = model.predict(pred_x)
    seed_sentence = seed_sentence[1:] + unique_chars[np.argmax(preds)]
    ss_copy = ss_copy + unique_chars[np.argmax(preds)]
print(ss_copy)


# Create a seed sentence.
seed_index = np.random.randint(len(text)-maxlen)
print("Seed index: {}".format(seed_index))

seed_sentence = text[seed_index:seed_index + maxlen]
ss_copy = text[seed_index:seed_index + maxlen]
print("Seed sentence: {}".format(seed_sentence))

for i in range(400):
    # Now to encode this sentence.
    pred_x = np.zeros((1, maxlen, len(unique_chars)), dtype=np.float32)
    for char_index, char in enumerate(seed_sentence):
        pred_x[0, char_index, char_to_idx[char]] = 1
        
    preds = model.predict(pred_x)[0]
#     preds = np.asarray(preds).astype(np.float32)
    preds = np.exp(preds)
    preds /= np.sum(preds)
    next_char = unique_chars[np.argmax(np.random.multinomial(1, preds, 1))]
    seed_sentence = seed_sentence[1:] + next_char
    ss_copy += next_char
    
print(ss_copy)


print(preds)
print(np.sum(preds))
preds = preds / np.sum(preds)

print(np.sum(preds))
