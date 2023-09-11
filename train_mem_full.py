# %%
import numpy as np
# import pandas as pd
# import sys
# import cv2
# from skimage.io import imread, imread_collection
# from scipy import signal
# import scipy.misc
# from scipy.misc import imread
# from matplotlib.pyplot import imread
# from cv2 import imread
from skimage.io import imread, imsave
from matplotlib import pyplot as plt

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"#-----specify device gpu or cpu

from data_gen_distance_class import data_gen
import tensorflow as tf
tf_version = int(tf.__version__[0])
from prep_data_distance_class import prep_data, prep_prediction_data
from deep_distance_class_estimator import deep_distance_class_estimator #, scheduler

# %% set folders
data_path=r'/media/data1/membrane_nucleus_segmentation_classification/Data_and_results'
path_input = os.path.join(data_path, 'img_mem')
path_dist = os.path.join(data_path, 'dist_mem')
path_class = os.path.join(data_path, 'edge_mem')
sub = 'predict-full'
path_weight = os.path.join(data_path, sub, 'weight_mem_full')
n_labels = 3

if not os.path.exists(path_weight):
    os.makedirs(path_weight)
# if not os.path.exists(path_output):
#     os.makedirs(path_output)

## complie model
autoencoder = deep_distance_class_estimator(n_labels = n_labels)
# autoencoder.summary()
losses = {'dist': "mean_squared_error",
        'class': "categorical_crossentropy"}
loss_weights = {'dist': 1,
        'class': 1}
autoencoder.compile(loss=losses, loss_weights=loss_weights, optimizer='adam', metrics=['mae','acc'])
print ('Compiled: OK')

# %% load images and labels
(all_data, all_label_dist, all_label_class, img_list) = prep_data(path_input,path_dist, path_class, n_labels)

# %%
for cv in [4]:
    print('Cross-validation round',cv)
    ## use all data in training
    train_data = all_data
    train_label_dist = all_label_dist
    train_label_class = all_label_class
    weight_file = os.path.join(path_weight, 'weight_EDT_class_{}.h5'.format(cv))


    # %%
    ## -----------------train--------------------------
    BATCH_SIZE = 8
    nb_epoch =1000
    NO_OF_TRAINING_IMAGES = train_data.shape[0]
    train_gen = data_gen(train_data, train_label_dist, train_label_class, batch_size=BATCH_SIZE, flips=True, rotate=True, intensity=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    # reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = autoencoder.fit_generator(train_gen, epochs=nb_epoch, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                            verbose=1, callbacks=[early_stop]) # , reduce_lr
    autoencoder.save_weights(weight_file)

    # %% 
    ## plot the training accuracy and loss at each epoch
    loss = history.history['loss']
    dist_loss = history.history['dist_loss']
    class_loss = history.history['class_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, dist_loss, 'y--', label='Training distance loss')
    plt.plot(epochs, class_loss, 'y:', label='Training class loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0,1])
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path_weight, 'EDT and class Training loss cv={}.png'.format(cv)))

    dist_acc = history.history['dist_acc']
    class_acc = history.history['class_acc']

    plt.figure()
    plt.plot(epochs, dist_acc, 'g--', label='Training Distance Accuracy')
    plt.plot(epochs, class_acc, 'g:', label='Training Class Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path_weight, 'EDT and class Training Accuracy cv={}.png'.format(cv)))
    plt.close('all')
