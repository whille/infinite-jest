# infinite-jest

A sequence generating model that uses David Foster Wallace's Infinite Jest as the input. If large datasets are needed for good results with deep learning, I can't think of a better candidate.

Trained on 4 Tesla V100s NVLINK, 15 vCPUs and 128GiB memory. 

There are a bunch of Keras/TF2 saved models:
- 60 epochs training a simple 64-unit LSTM with Adam and a learning rate of 0.01. About a minute per epoch, with a minibatch size of 128. Based on the Keras guide [here](https://keras.io/examples/generative/lstm_character_level_text_generation/).

To do:
- [x] Get `CuDNNLSTM` layers instead of plain `LSTM` layers to take advantage of the GPUs.
    - Actually, Tensorflow uses the CUDA accelerated version automatically if the definition of the layer satisfies some set of parameters.
