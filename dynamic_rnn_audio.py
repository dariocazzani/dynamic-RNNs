"""
Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length.
"""

import tensorflow as tf
import random, functools
import numpy as np
from noise_generator import *

_COLORS = ['white', 'pink', 'blue', 'brown', 'violet']

def compute_mel_spectrogram(audio):
	# NB: Not clear in the API, input to stft needs to be float32
	_WINLEN = 0.025
	_STEP_SIZE = 0.01
	_FBANK_COEFF = 64
	_SAMPLE_RATE = 16000
	audio = tf.cast(audio, tf.float32)
	stfts = tf.contrib.signal.stft(
				audio,
				int(_WINLEN*_SAMPLE_RATE),
				int(_STEP_SIZE*_SAMPLE_RATE),
				512,
				window_fn=functools.partial(tf.contrib.signal.hann_window, periodic=True),
				pad_end=True)
	spectrograms = tf.abs(stfts)

	# Warp the linear scale spectrograms into the mel-scale.
	num_spectrogram_bins = stfts.shape[-1].value
	lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, _SAMPLE_RATE/2, _FBANK_COEFF
	linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
					num_mel_bins, num_spectrogram_bins, _SAMPLE_RATE, lower_edge_hertz, upper_edge_hertz)
	mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
	mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

	return mel_spectrograms

def pad_audio(audio, MIN_LENGTH):
	if len(audio) < MIN_LENGTH:
		audio = np.lib.pad(audio, (0, MIN_LENGTH - len(audio)), 'constant', constant_values=(0))
	return audio

class AudioGenerator(object):
    def __init__(self, max_seq_len=32000, min_seq_len=16000):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.max_seq_len=max_seq_len
        self.min_seq_len=min_seq_len

    def next(self, batch_size):
        for i in range(batch_size):
            # Random sequence length
            length = random.randint(self.min_seq_len, self.max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(length)

            # Generate noise
            color = random.sample(_COLORS, 1)[0]
            noise = globals()[color](length)
            noise = pad_audio(noise, self.max_seq_len)

            # Generate label
            index = _COLORS.index(color)
            one_hot = np.zeros(len(_COLORS))
            one_hot[index] = 1

            # Make it into a list:
            self.data.append(list(noise))
            self.labels.append(list(one_hot))

        batch_data = self.data.copy()
        batch_labels = self.labels.copy()
        batch_seqlen = self.seqlen.copy()
        self.data = []
        self.labels = []
        self.seqlen = []
        return np.array(batch_data), batch_labels, batch_seqlen


# Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 1

# Network Parameters
seq_max_len = 32000 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 5 # what noise color is this?
vector_length = 64 # Number of MFC coefficients

trainset = AudioGenerator(max_seq_len=seq_max_len, min_seq_len=16000)
testset = AudioGenerator(max_seq_len=seq_max_len, min_seq_len=16000)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len], name='x')
y = tf.placeholder("float", [None, n_classes], name='y')
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None], name='seqlen')

def dynamicRNN(x, seqlen):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, np.ceil(seq_max_len/160), 1)
    seqlen = tf.cast(tf.ceil(seqlen / 160), tf.int32)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * int(seq_max_len/160) + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.layers.dense(outputs, n_classes)

preprocessed_audio = compute_mel_spectrogram(x)
pred = dynamicRNN(preprocessed_audio, seqlen)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    try:
        for step in range(1, training_steps + 1):
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                    seqlen: batch_seqlen})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    except (KeyboardInterrupt, SystemExit):
        print("Manual interrupt")

    print("Optimization Finished!")

    # Calculate accuracy
    test_data, test_label, test_seqlen = testset.next(50)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
