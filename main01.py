# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:16:31 2025

@author: joshua
"""

# start the spyder kernel in wsl:
# conda activate tfgpu
# python -m spyder_kernels.console
# python -m spyder_kernels.console --f=/home/joshua/.local/share/jupyter/runtime/my_kernel.json

# access it using the Consoles > existing kernel:
# //wsl$/Ubuntu/home/joshua/.local/share/jupyter/runtime/my_kernel.json


# %% imports

import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers, losses
# import keras.utils as np_utils
from keras.utils import to_categorical
from keras.datasets import mnist

import matplotlib.pyplot as plt

# import wandb
# from wandb.integration.keras import WandbCallback


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
noise_dim = 20

batch_size = 512
C_epochs = 25
DG_epochs = 100
steps_per_epoch = x_train.shape[0] // batch_size

learning_rate = 2e-3
beta_1 = 0.5
beta_2 = 0.999


# %% setup wandb

# wandb.init(
#     project = 'MNIST-Generative-Model',
#     name = 'run-01',
#     config=dict(
#         batch_size = batch_size,
#         epochs = DG_epochs,
#         lr = learning_rate,
#         beta1 = beta_1,
#         beta2 = beta_2,
#         notes = 'Keras model, MNIST image AC-GAN, 9 metrics')
#     )


# %% models
print('setup models.')

def build_classifier(img_shape=(28, 28, 1), num_classes=10):
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
    
    # branch for class
    output_class = layers.Dense(num_classes, activation='softmax', name='class')(x)
    
    return models.Model(img_input, output_class, name='classifier')

def build_descriminator(img_shape=(28, 28, 1)):
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
    
    output_rf = layers.Dense(1, activation='sigmoid', name='real_fake')(x)
    
    return models.Model(img_input, output_rf, name='descriminator')

def build_generator(noise_dim=100, num_classes=10):
    # build noise, embedding of the classes, and concatenate into the first layer
    noise_input = layers.Input(shape=(noise_dim,), name='noise')
    class_input = layers.Input(shape=(num_classes,), dtype='int32', name='label')
    class_emb = layers.Embedding(num_classes, 5)(class_input)
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

def build_generator_trainer(noise_dim=100, num_classes=10):
    noise_input = layers.Input(shape=(noise_dim,))
    class_input = layers.Input(shape=(num_classes,), dtype='int32')
    
    fake_img = G([noise_input, class_input])
    
    class_pred = C(fake_img)
    rf_pred = D(fake_img)
    
    return models.Model([noise_input, class_input], [rf_pred, class_pred], name='G_CD_stacked')

G = build_generator(noise_dim, num_classes)
C = build_classifier(img_shape, num_classes)
D = build_descriminator(img_shape)
G_CD = build_generator_trainer(noise_dim, num_classes)


# %% optimise and compile
print('setup optimisers and compile.')

# compile classifier
cls_opt = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
C.compile(
    optimizer = cls_opt,
    loss = {'class': losses.CategoricalCrossentropy()},
    loss_weights = {'class': 1.0},
    metrics = {'class': 'accuracy'}
    )

# compile descriminator
des_opt = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
D.compile(
    optimizer = des_opt,
    loss = {'real_fake': losses.BinaryCrossentropy(from_logits=False)},
    loss_weights = {'real_fake': 1.0},
    metrics = {'real_fake': 'accuracy'}
    )

# compile generator training combined model
gen_opt = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
G_CD.compile(
    optimizer=gen_opt,
    loss={
        'descriminator': losses.BinaryCrossentropy(from_logits=False),
        'classifier': losses.CategoricalCrossentropy()
        },
    loss_weights={
        'descriminator': 1.0,
        'classifier': 1.0
        },
    metrics={
        'descriminator': 'accuracy',
        'classifier': 'accuracy'
        }
    )

# %% generate image

def gen_digit(digit:int, epoch=None):
    noise_input = np.random.normal(0, 1, (1, noise_dim))
    class_input = tf.keras.utils.to_categorical([digit], num_classes)
    fig = G.predict([noise_input, class_input], verbose=0)[0, :, :, 0]
    return fig

def plot_digit(image, label, epoch=None, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(f"MNIST digit:{label}" if epoch == None else f"MNIST digit:{label} - epoch:{epoch}")
    plt.axis('off')
    plt.show()
    return

def plot_20_digits(epoch_imgs, cmap='gray'):
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for ax, (img, digit, epoch) in zip(axes.ravel(), epoch_imgs):
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"digit:{digit}" if epoch == None else f"digit:{digit} - epoch:{epoch}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    return

def make_unique(xs):
    """Round to int and increment duplicates until all are unique."""
    out = []
    seen = set()
    for x in np.round(xs).astype(int):
        while x in seen:
            x += 1
        seen.add(x)
        out.append(x)
    return np.array(out)

# prepare 4x5 plots for after all images generated
epoch_imgs = []
n_imgs = 20 if DG_epochs > 20 else DG_epochs
if n_imgs < 20:
    idx_imgs = np.round(np.linspace(1, n_imgs-1, n_imgs)).astype(int)
else:
    xs = np.logspace(0, np.log10(DG_epochs), num=n_imgs, endpoint=True)
    idx_imgs = make_unique(xs)


# %% train
print('Begin training:')

fixed_noise = np.random.randn(num_classes, noise_dim).astype('float32')
fixed_class = np.arange(num_classes, dtype='int32')

# descriminator headstart training
print('Classifier:')
C.trainable = True
for epoch in range(1, C_epochs+1):
    for step in range(steps_per_epoch):
        # train descriminator
        # idx_r = np.random.randint(0, x_train.shape[0], batch_size)
        idx_r = tf.random.uniform([batch_size], maxval=x_train.shape[0], dtype=tf.int32)
        real_imgs = tf.gather(x_train, idx_r)
        real_imgs = tf.expand_dims(real_imgs, axis=-1) # expand the missing dimension
        real_labels = tf.gather(y_train, idx_r)
        
        c_train = C.train_on_batch(real_imgs, real_labels)
        
    # log data each epoch
    print('epoch {:02d}/{} -'.format(epoch, C_epochs), 
          'C_loss: {:.3f}\tC_acc: {:.3f}'.format(c_train[0], c_train[1]))

# descriminator and generator training
print('Descriminator and Generator:')
C.trainable = False
for epoch in range(1, DG_epochs+1):
    for step in range(steps_per_epoch):
        
        # train descriminator
        # real batch
        idx_r = tf.random.uniform([batch_size], maxval=x_train.shape[0], dtype=tf.int32)
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
        G.trainable = False
        d_loss_real = D.train_on_batch(real_imgs, y_rf_real)
        d_loss_fake = D.train_on_batch(fake_imgs, y_rf_fake)
        
        
        # train generator (via combined model)
        idx_c = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)
        cond_labels = tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32)
        cond_labels = to_categorical(cond_labels, num_classes) # convert to one-hot
        y_rf_trick  = tf.ones([batch_size, 1], dtype=tf.float32)
        
        D.trainable = False
        G.trainable = True
        g_loss = G_CD.train_on_batch([idx_c, cond_labels], [y_rf_trick, cond_labels])
        
        pass # TODO: breakpoint for testing
        
        
    # log data each epoch
    print('epoch {:02d}/{} -'.format(epoch, DG_epochs), 
          'D_r/f_loss: {:.3f}/{:.3f}\tD_r/f_acc: {:.3f}/{:.3f}\t'.format(d_loss_real[0], d_loss_fake[0], d_loss_real[1], d_loss_fake[1]), 
          'G_total/d/c_loss: {:.3f}/{:.3f}/{:.3f}\tG_d/c_acc: {:.3f}/{:.3f}'.format(g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4]))
    
    # generate a random digit for visual progress
    digit = np.random.randint(0, num_classes)
    fig = gen_digit(digit, epoch)
    if epoch in idx_imgs: epoch_imgs.append((fig, digit, epoch))
    plot_digit(fig, digit, epoch)
    
plot_20_digits(epoch_imgs)
    

# %% Plot diagram

# from tensorflow.keras.utils import plot_model

# # Full stacked model with submodels expanded
# plot_model(
#     G_CD,
#     to_file="G_CD.png",
#     show_shapes=True,
#     show_layer_names=True,
#     expand_nested=True,   # <- shows G, D, C internals
#     rankdir="LR",         # left-to-right; use "TB" for top-to-bottom
#     dpi=220
# )

# # # Optional: also export components
# # plot_model(G,  to_file="G.png",  show_shapes=True, rankdir="LR", dpi=200)
# # plot_model(D,  to_file="D.png",  show_shapes=True, rankdir="LR", dpi=200)
# # plot_model(C,  to_file="C.png",  show_shapes=True, rankdir="LR", dpi=200)

