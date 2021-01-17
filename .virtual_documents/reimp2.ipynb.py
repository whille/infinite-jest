import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) == 0:
    print("You're not GPU-enabled")
else:
    for gpu in gpus:
        print(gpu)


path = "infinite_jest_text.txt"

with open(path, "r") as f:
    text = f.read()
    
text = text.lower().replace("\n", " ")
chars = sorted(list(set(text)))

# We need to be able to convert from chars to indices and vice-versa.
char_to_idx = dict((c, i) for (i, c) in enumerate(chars))
idx_to_char = dict((i, c) for (i, c) in enumerate(chars))


sentences, next_chars = [], []
stride = 6
sentlen = 40

# Store sentences and the character that follows each of them.
for i in range(0, len(text)-sentlen, stride):
    sentences.append(text[i : i+sentlen])
    next_chars.append(text[i+sentlen])
    
# These will be our inputs and outputs.
x = np.zeros((len(sentences), sentlen, len(chars)), dtype=np.integer)
y = np.zeros((len(sentences), len(chars)), dtype=np.integer)
    
for sent_index, sentence in enumerate(sentences):
    for char_index, char in enumerate(sentence):
        x[sent_index, char_index, char_to_idx[char]] = 1 # One-hot encode each character of the sentence.
    y[sent_index, char_to_idx[next_chars[sent_index]]] = 1 # One-hot encode the next character.


def sentence_from_encoding(sent_encoding):
    sent = []
    for char_encoding in sent_encoding:
        sent.append(chars[np.argmax(char_encoding)])
        
    return "".join(sent)


print(x[0])
print(sentence_from_encoding(x[0]))


model = keras.Sequential(
[
    keras.layers.Input((sentlen, len(chars)), name="Input01"),
    keras.layers.LSTM(128, return_sequences=True, name="LSTM01"),
    keras.layers.Dropout(0.8, name="Dropout01"),
    keras.layers.LSTM(64, return_sequences=True, name="LSTM02"),
    keras.layers.Dropout(0.8, name="Dropout02"),
    keras.layers.LSTM(32, name="LSTM03"),
    keras.layers.Dropout(0.8, name="Dropout03"),
    keras.layers.Dense(len(chars), activation="softmax", name="Softmax")
])


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(x, y, epochs=1, batch_size=128)


seed_index = np.random.randint(len(text)-sentlen)
seed_sentence = text[seed_index : seed_index + sentlen]
ss_copy = text[seed_index : seed_index + sentlen]

for i in range(400):
    # Encode seed sentence.
    x_pred = np.zeros((1, sentlen, len(chars)), dtype=np.integer)
    for char_index, char in enumerate(seed_sentence):
        x_pred[0, char_index, char_to_idx[char]] = 1
        
    preds = model.predict(x_pred)[0]
    
    # Re-normalise predictions to avoid numerical underflow issues
    # with sampling from it.
    preds = np.exp(preds)
    preds = preds / np.sum(preds)
    next_char = idx_to_char[np.argmax(np.random.multinomial(1, preds, 1))]
    
    seed_sentence = seed_sentence[1:] + next_char
    ss_copy = ss_copy + next_char


print(ss_copy)
