#! /usr/local/bin/python
#
# Machine Learning - Assignment 3
# Astegher Maurizio 175195

import input_data
import tensorflow as tf

# Init for a weight variable
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# Init for a bias variable
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Values that we will input during the computation
x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])

# Reshape vectors of size 784 to squares of size 28x28
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Layer 1: convolutional layer with max pooling 
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

# Layer 2: convolutional layer with max pooling 
# W_conv2 = weight_variable([5, 5, 32, 64])
W_conv2 = weight_variable([5, 5, 1, 64])
b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2 = tf.nn.relu(conv2d(x_image, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Layer 3: ReLU & dropout
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
W_fc1 = weight_variable([14 * 14 * 64, 1024])
b_fc1 = bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool2,[-1, 7 * 7 * 64])
h_pool2_flat = tf.reshape(h_pool2,[-1, 14 * 14 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Layer 4: softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_hat = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define the cost: cross-entropy
cross_entropy = -tf.reduce_sum(y * tf.log(y_hat))

# Define the training algorithm 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Start a new session
sess = tf.InteractiveSession()

# Initialize the variables
sess.run(tf.initialize_all_variables())

# Define the accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Train the model
for n in range(20000):
	batch = mnist.train.next_batch(50)
	if n % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
		print "step %d, training accuracy %g" % (n, train_accuracy)
	sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})

# Evaluate the prediction
print "test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
