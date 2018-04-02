"""
Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length.
"""

import tensorflow as tf
import random
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
class MnistGenerator(object):
    def __init__(self, max_seq_len=28, min_seq_len=20):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.max_seq_len=max_seq_len
        self.min_seq_len=min_seq_len

    def next(self, batch_size):
        for i in range(batch_size):
            sample_x, sample_y = mnist.train.next_batch(1)
            # Random sequence length
            length = random.randint(self.min_seq_len, self.max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(length)

            # reshape to a sequence of vectors
            sample_x = np.reshape(sample_x, [28, 28])
            sample_y = np.squeeze(sample_y)

            # Zero out anything from seqlen until the end of the image
            sample_x[:, length:] = 0.

            # Make it into a list:
            self.data.append(list(sample_x))
            self.labels.append(list(sample_y))

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
display_step = 200

# Network Parameters
seq_max_len = 28 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 10 # what digit the input is
vector_length = 28 #

trainset = MnistGenerator(max_seq_len=seq_max_len, min_seq_len=20)
testset = MnistGenerator(max_seq_len=seq_max_len, min_seq_len=20)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, vector_length], name='x')
y = tf.placeholder("float", [None, n_classes], name='y')
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None], name='seqlen')

def dynamicRNN(x, seqlen):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

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
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.layers.dense(outputs, n_classes)

pred = dynamicRNN(x, seqlen)

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
    test_data, test_label, test_seqlen = testset.next(5000)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
