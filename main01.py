# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:16:31 2025

@author: joshua
"""

# start the spyder kernel in wsl:
# conda activate aigpu
# python -m spyder_kernels.console

# access it using the Consoles > existing kernel:
# //wsl$/Ubuntu/home/joshua/.local/share/jupyter/runtime/kernel-****.json

# %% imports

import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers, losses
import keras.utils as np_utils
from keras.utils import to_categorical
from keras.datasets import mnist

import matplotlib.pyplot as plt


# %% setup gpu support

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Allow memory growth (prevents TF from grabbing all GPU RAM)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# %% Load Data
print('load data.')

# download mnist data
(x_train, y_train), (_, _) = mnist.load_data()

# Normalise x data (features)
# x_train = x_train.astype('float32') / 255
x_train = tf.cast(x_train, tf.float32) / 255.0

# one-hot the y data (categories)
# y_train = np_utils.to_categorical(y_train)
num_classes = len(np.unique(y_train))
y_train = tf.one_hot(y_train, depth=num_classes)


# %% Config
print('set config.')

img_shape = (28, 28, 1)
noise_dim = 100

batch_size = 512
D_epochs = 5
G_epochs = 10
DG_epochs = 35
steps_per_epoch = x_train.shape[0] // batch_size

learning_rate = 2e-4
beta_1 = 0.5
beta_2 = 0.999

# %% models
print('setup models.')

# Takes noise_dim which is an added noise of n dimensions, num_classes is the number of classes which will be shaped into a size 50 embedding before concatenating with the noise
def build_generator(noise_dim=100, num_classes=10):
    # build noise, embedding of the classes, and concatenate into the first layer
    noise_input = layers.Input(shape=(noise_dim,), name='noise')
    class_input = layers.Input(shape=(num_classes,), dtype='int32', name='label')
    class_emb = layers.Embedding(num_classes, 50)(class_input)
    class_emb = layers.Flatten()(class_emb)
    x = layers.Concatenate()([noise_input, class_emb])
    
    # dense layer, size is 7*7*256 for a 7x7 image with 256 channels, to provide for later convolution steps
    x = layers.Dense(7*7*256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((7, 7, 256))(x)

    # convolution layers
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # output layer, 
    img_output = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(x)
    
    return models.Model([noise_input, class_input], img_output, name='generator')

def build_descriminator(img_shape=(28, 28, 1), num_classes=10):
    img_input = layers.Input(shape=img_shape)
    
    x = layers.Conv2D(64, 3, strides=2, padding='same')(img_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    
    # branch for real/false doesn't affect convolution layers
    # x_rf = layers.Lambda(lambda t: tf.stop_gradient(t), name="detach_for_rf")(x)
    # output_rf = layers.Dense(1, activation='sigmoid', name='real_fake')(x_rf)
    output_rf = layers.Dense(1, activation='sigmoid', name='real_fake')(x)
    
    # branch for class
    output_class = layers.Dense(num_classes, activation='softmax', name='class')(x)
    
    return models.Model(img_input, [output_rf, output_class], name='descriminator')

G = build_generator(noise_dim, num_classes)
D = build_descriminator(img_shape, num_classes)


# %% optimise and compile
print('setup optimisers and compile.')

gen_opt = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
des_opt = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

# compile descriminator
D.compile(
    optimizer = des_opt,
    loss = {
        'real_fake': losses.BinaryCrossentropy(from_logits=False),
        'class': losses.CategoricalCrossentropy()
        },
    loss_weights = {'real_fake': 0.1, 'class': 1.0},
    metrics = {'real_fake': 'accuracy', 'class': 'accuracy'}
    )

# combine models (for generator learning)
noise_input = layers.Input(shape=(noise_dim,))
class_input = layers.Input(shape=(num_classes,), dtype='int32')
fake_img = G([noise_input, class_input])
D.trainable = False
rf_pred, class_pred = D(fake_img)
combined = models.Model([noise_input, class_input], [rf_pred, class_pred], name='G_D_stacked')

# compile the combined model
combined.compile(
    optimizer = gen_opt,
    loss = [
        losses.BinaryCrossentropy(from_logits=False),
        losses.CategoricalCrossentropy()
        ],
    loss_weights = [1.0, 1.0]
    )


# %% generate image

def plot_digit(image, label, epoch=None, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(f"MNIST digit:{label}" if epoch != None else f"MNIST digit:{label} - epoch:{epoch}")
    plt.axis('off')
    plt.show()
    return

def gen_digit(digit:int, epoch=None):
    noise_input = np.random.normal(0, 1, (1, noise_dim))
    class_input = tf.keras.utils.to_categorical([digit], num_classes)
    plot_digit(G.predict([noise_input, class_input], verbose=0)[0, :, :, 0], digit, epoch)
    return


# %% train
print('train.')

fixed_noise = np.random.randn(num_classes, noise_dim).astype('float32')
fixed_class = np.arange(num_classes, dtype='int32')

# descriminator headstart training
print('Descriminator only:')
for epoch in range(1, D_epochs+1):
    for step in range(steps_per_epoch):
        # train descriminator
        # real batch
        idx_r = np.random.randint(0, x_train.shape[0], batch_size)
        # idx_r = tf.random.uniform([batch_size], maxval=x_train.shape[0], dtype=tf.int32)
        real_imgs = tf.gather(x_train, idx_r)
        real_imgs = tf.expand_dims(real_imgs, axis=-1) # expand the missing dimension
        real_labels = tf.gather(y_train, idx_r)
        y_rf_real = tf.ones([batch_size, 1], dtype=tf.float32)
        
        # # fake batch
        # idx_f = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)
        # fake_labels = tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32)
        # fake_labels = to_categorical(fake_labels, num_classes) # convert to one-hot
        # fake_imgs = G([idx_f, fake_labels], training=True)
        # y_rf_fake = tf.zeros([batch_size, 1], dtype=tf.float32)
        
        # train seperately for stability
        D.trainable = True
        d_loss_real = D.train_on_batch(real_imgs, [y_rf_real, real_labels])
        # d_loss_fake = D.train_on_batch(fake_imgs, [y_rf_fake, fake_labels])
        
    # log data each epoch
    print(f'epoch {epoch:02d}/{D_epochs} - '
          f'D_real_acc (rf/cls): {d_loss_real[3]:.3f}/{d_loss_real[4]:.3f}')
          # f'D_fake_acc (rf/cls): {d_loss_fake[3]:.3f}/{d_loss_fake[4]:.3f}')

# generator only training
print('Generator only:')
for epoch in range(1, G_epochs+1):
    for step in range(steps_per_epoch):
        # train generator (via combined model)
        D.trainable = False
        idx_c = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)
        cond_labels = tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32)
        cond_labels = to_categorical(cond_labels, num_classes) # convert to one-hot
        y_rf_trick  = tf.ones([batch_size, 1], dtype=tf.float32)
        g_loss = combined.train_on_batch([idx_c, cond_labels], [y_rf_trick, cond_labels])
        
    # log data each epoch
    print(f'epoch {epoch:02d}/{G_epochs} - '
          f'G_loss: {g_loss if isinstance(g_loss, float) else g_loss[0]:.3f}')
    
    # generate a random digit for visual progress
    gen_digit(np.random.randint(0, num_classes), epoch)

# descriminator and generator training
print('Descriminator and Generator')
for epoch in range(1, DG_epochs+1):
    for step in range(steps_per_epoch):
        # train descriminator
        # real batch
        idx_r = np.random.randint(0, x_train.shape[0], batch_size)
        # idx_r = tf.random.uniform([batch_size], maxval=x_train.shape[0], dtype=tf.int32)
        real_imgs = tf.gather(x_train, idx_r)
        real_imgs = tf.expand_dims(real_imgs, axis=-1) # expand the missing dimension
        real_labels = tf.gather(y_train, idx_r)
        y_rf_real = tf.ones([batch_size, 1], dtype=tf.float32)
        
        # fake batch
        idx_f = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)
        fake_labels = tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32)
        fake_labels = to_categorical(fake_labels, num_classes) # convert to one-hot
        fake_imgs = G([idx_f, fake_labels], training=True)
        y_rf_fake = tf.zeros([batch_size, 1], dtype=tf.float32)
        
        # train seperately for stability
        D.trainable = True
        d_loss_real = D.train_on_batch(real_imgs, [y_rf_real, real_labels])
        d_loss_fake = D.train_on_batch(fake_imgs, [y_rf_fake, fake_labels])
        
        # train generator (via combined model)
        D.trainable = False
        idx_c = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)
        cond_labels = tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32)
        cond_labels = to_categorical(cond_labels, num_classes) # convert to one-hot
        y_rf_trick  = tf.ones([batch_size, 1], dtype=tf.float32)
        g_loss = combined.train_on_batch([idx_c, cond_labels], [y_rf_trick, cond_labels])
        
    # log data each epoch
    print(f'epoch {epoch:02d}/{DG_epochs} - '
          f'D_real_acc (rf/cls): {d_loss_real[3]:.3f}/{d_loss_real[4]:.3f} - '
          f'D_fake_acc (rf/cls): {d_loss_fake[3]:.3f}/{d_loss_fake[4]:.3f} - '
          f'G_loss: {g_loss if isinstance(g_loss, float) else g_loss[0]:.3f}')
    
    # generate a random digit for visual progress
    gen_digit(np.random.randint(0, num_classes), epoch)


# %% plotting

# # Plot accuracy
# plt.plot(history.history['accuracy'], label='train acc')
# plt.plot(history.history['val_accuracy'], label='val acc')
# plt.title('CNNModel Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Plot loss
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.title('CNNModel Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

