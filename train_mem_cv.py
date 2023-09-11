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
import scipy.io as sio

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"#-----specify device gpu or cpu

from data_gen_distance_class import data_gen
import tensorflow as tf
tf_version = int(tf.__version__[0])
from tensorflow.keras.metrics import MeanIoU
import keras.backend as K
from prep_data_distance_class import prep_data, prep_prediction_data
from deep_distance_class_estimator import deep_distance_class_estimator #, scheduler

# %% set folders
data_path=r'/media/data1/membrane_nucleus_segmentation_classification/Data_and_results'
path_input = os.path.join(data_path, 'img_mem')
path_dist = os.path.join(data_path, 'dist_mem')
path_class = os.path.join(data_path, 'edge_mem')
sub = 'cross-validation'
path_weight = os.path.join(data_path, sub, 'weight_mem_cv')
path_EDT = os.path.join(data_path, sub, 'EDT_mem_cv')
path_output = os.path.join(data_path, sub, 'output_mem_cv')
n_labels = 3

if not os.path.exists(path_weight):
    os.makedirs(path_weight)
if not os.path.exists(path_EDT):
    os.makedirs(path_EDT)
if not os.path.exists(path_output):
    os.makedirs(path_output)

# %% load images and labels
(all_data, all_label_dist, all_label_class, img_list) = prep_data(path_input,path_dist,path_class, n_labels)

# %% 
## set groups for 4-fold cross-validation
group = np.zeros((4,8*8//4),dtype='uint16')
group1 = np.zeros((8,8),dtype='bool')
group1[0:4,0:4] = 1
group[0] = group1.ravel().nonzero()[0]
group1 = np.zeros((8,8),dtype='bool')
group1[0:4,4:8] = 1
group[1] = group1.ravel().nonzero()[0]
group1 = np.zeros((8,8),dtype='bool')
group1[4:8,0:4] = 1
group[2] = group1.ravel().nonzero()[0]
group1 = np.zeros((8,8),dtype='bool')
group1[4:8,4:8] = 1
group[3] = group1.ravel().nonzero()[0]

# %%
for cv in range(4):
    ## complie model
    autoencoder = deep_distance_class_estimator(n_labels = n_labels)
    # autoencoder.summary()
    losses = {'dist': "mean_squared_error",
            'class': "categorical_crossentropy"}
    loss_weights = {'dist': 1,
            'class': 1}
    autoencoder.compile(loss=losses, loss_weights=loss_weights, optimizer='adam', metrics=['mae','acc'])
    print ('Compiled: OK')

    print('Cross-validation round',cv)
    ## divide training and validation (test) data
    testset = group[cv]
    trainset = group[list(range(0,cv)) + list(range(cv+1,4))].ravel()
    train_data = all_data[trainset]
    train_label_dist = all_label_dist[trainset]
    train_label_class = all_label_class[trainset]
    test_data = all_data[testset]
    test_label_dist = all_label_dist[testset]
    test_label_class = all_label_class[testset]
    test_name = [img_list[x] for x in testset]
    weight_file = os.path.join(path_weight, 'weight_EDT_class_{}.h5'.format(cv))


    # %%
    ## -----------------train--------------------------
    BATCH_SIZE = 8
    nb_epoch =1000
    NO_OF_TRAINING_IMAGES = train_data.shape[0]
    NO_OF_VAL_IMAGES = test_data.shape[0]
    train_gen = data_gen(train_data, train_label_dist, train_label_class, batch_size=BATCH_SIZE, flips=True, rotate=True, intensity=False)
    val_gen = data_gen(test_data, test_label_dist, test_label_class, batch_size=NO_OF_VAL_IMAGES, flips=False, rotate=False, intensity=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    # reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = autoencoder.fit_generator(train_gen, epochs=nb_epoch, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                            validation_data=val_gen, validation_steps=1, verbose=1, callbacks=[early_stop]) # , reduce_lr
    autoencoder.save_weights(weight_file)

    # %% 
    ## plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    dist_loss = history.history['dist_loss']
    class_loss = history.history['class_loss']
    val_loss = history.history['val_loss']
    val_dist_loss = history.history['val_dist_loss']
    val_class_loss = history.history['val_class_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, dist_loss, 'y--', label='Training distance loss')
    plt.plot(epochs, class_loss, 'y:', label='Training class loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.plot(epochs, val_dist_loss, 'r--', label='Validation distance loss')
    plt.plot(epochs, val_class_loss, 'r:', label='Validation class loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0,1])
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path_weight, 'EDT and class Training and validation loss cv={}.png'.format(cv)))

    dist_acc = history.history['dist_acc']
    class_acc = history.history['class_acc']
    val_dist_acc = history.history['val_dist_acc']
    val_class_acc = history.history['val_class_acc']

    plt.figure()
    plt.plot(epochs, dist_acc, 'g--', label='Training Distance Accuracy')
    plt.plot(epochs, class_acc, 'g:', label='Training Class Accuracy')
    plt.plot(epochs, val_dist_acc, 'm--', label='Validation Distance Accuracy')
    plt.plot(epochs, val_class_acc, 'm:', label='Validation Class Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path_weight, 'EDT and class Training and validation Accuracy cv={}.png'.format(cv)))
    plt.close('all')

    # %% 
    ## -----------predict--------------
    autoencoder.load_weights(weight_file)
    # (test_data,img_list) = prep_prediction_data(path_predict)
    (dist, classes) = autoencoder.predict(test_data, verbose=0)
    print(type(dist))
    print(dist.shape)
    print(type(classes))
    print(classes.shape)

    n_test = len(test_name)
    list_values = np.zeros((n_test, n_labels, n_labels))
    list_mean_IoU = np.zeros(n_test)
    list_class_IoU = np.zeros((n_test, n_labels))

    for ind in range(n_test):
        ## save predicted distance
        img = dist[ind,:,:,0]
        dir_save = test_name[ind].replace('img','EDT')
        imsave(os.path.join(path_EDT, dir_save), img)

        ## save predicted classes. The numbers are multiplied by 127 to enlarge image contrast.
        img = np.argmax(classes[ind],axis=-1).astype('uint8')
        dir_save = test_name[ind].replace('img','class')
        imsave(os.path.join(path_output, dir_save), img*127)

        # %% calculate mean IoU
        IOU_keras = MeanIoU(num_classes=n_labels)  
        IOU_keras.update_state(np.argmax(test_label_class[ind],-1), img)
        mean_IoU = IOU_keras.result().numpy()
        print("Mean IoU =", mean_IoU)
        ## To calculate I0U for each class
        values = np.array(IOU_keras.get_weights()).reshape(n_labels, n_labels)
        print(values)
        list_values[ind] = values
        list_mean_IoU[ind] = mean_IoU
        for classn in range(n_labels):
            classn_IoU = values[classn,classn] / (values[classn,:].sum() + values[:,classn].sum() - values[classn,classn])
            print("IoU for class {} is: {}".format(classn, classn_IoU))
            list_class_IoU[ind,classn] = classn_IoU
    
    # %% 
    ## save IoU information for all test data in this round of cross validation.
    sio.savemat(os.path.join(path_output, 'MeanIoU_cv{}.mat'.format(cv)), {"values": list_values, "class_IoU": list_class_IoU, "mean_IoU": list_mean_IoU})


