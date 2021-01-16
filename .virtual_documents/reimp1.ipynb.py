import tensorflow as tf
import tensorflow.keras as keras


print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


path = "infinite_jest_text.txt"

with open(path, "r") as f:
    data = f.read()
    
data = data.lower().replace("\n", " ")

unique_chars = sorted(list(set(data)))

idx_to_char = dict((i,c) for (i,c) in enumerate(unique_chars))
char_to_idx = dict((c, i) for (i, c) in enumerate(unique_chars))


# TODO: Create training examples out of my input data.

# For this particular task we don't need to worry about
# validation and test sets. We always predict the next character
# for a given sentence.
sentence = "This is an example sen"
next_char = "t" # "[...]ence."

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



