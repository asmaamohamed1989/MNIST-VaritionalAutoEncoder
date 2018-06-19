"""
This is a tutorial code followed from Felix Mohhr on :  https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776
"""


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# Downloading the MNIST dataset using tensorflow helper
from tensorflow.examples.tutorials.mnist import input_data
mnist_dataset = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()

# Define the size of the miniBatch (64 images)
batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')


Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])

dec_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_channels]
inputs_decoder = 49 * dec_channels / 2

# Defining leaky Relu function
def lrelu(x, alpha=0.3):
    return tf.arg_max(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
    """
    :param X_in: data input
    :param keep_prob: dropout variable
    :return:
    """
    activation = tf.nn.leaky_relu

    with tf.variable_scope("encoder", reuse=None):
        # reshape input tensor to matrix with one color channel
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)

        # flatten the last layer (ie make it a vector)
        x = tf.contrib.layers.flatten(x)
        mean = tf.layers.dense(x, units=n_latent)
        std = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))

        # create the encoded representation of the input X_in
        z = mean + tf.multiply(epsilon, tf.exp(std))

        return z, mean, std



def decoder(sampled_z, keep_prob):
    """
    :param sampled_z: Encoded version in the latent_space by the encoder network
    :param keep_prob: dropout variable
    :return: decoded version of the image
    """
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=tf.nn.leaky_relu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=tf.nn.leaky_relu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img



sampled, mean, std = encoder(X_in, keep_prob)
reconstructed_img = decoder(sampled, keep_prob)

# reshape the reconstruded image as a vector
flat_decode_img = tf.reshape(reconstructed_img, shape=[-1, 28*28])

img_loss = tf.reduce_sum(tf.squared_difference(flat_decode_img, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * std - tf.square(mean) - tf.exp(2.0 * std), 1)
total_loss = tf.reduce_mean(img_loss + latent_loss)

# define the optimization algorithm
learning_rate = 0.0005
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

sess = tf.Session()

# initialize the session and variables attach to it
sess.run(tf.global_variables_initializer())


iterations = 600
for i in range(iterations):
    # create the mini batch
    batch = [np.reshape(b, [28, 28]) for b in mnist_dataset.train.next_batch(batch_size=batch_size)[0]]
    sess.run(optimizer, feed_dict={X_in : batch, Y : batch, keep_prob : 0.8})

    if not i % 200:
        returnTotal_loss, d, returnImg_loss, returnLatent_loss, mu, sigm = sess.run([total_loss, reconstructed_img, img_loss, latent_loss, mean, std],
                                               feed_dict={X_in: batch, Y: batch, keep_prob: 1.0})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, returnTotal_loss, np.mean(returnImg_loss), np.mean(returnLatent_loss))

tf.train.write_graph(tf.get_default_graph(), ".", "vae_model.pb", as_text=False)

# use the trained network to generate image:
# sample from normal distribution
randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
# feed the decoder part of the network with the samples and perform a forward pass
imgs = sess.run(reconstructed_img, feed_dict = {sampled: randoms, keep_prob: 1.0})

# display the generated images
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')