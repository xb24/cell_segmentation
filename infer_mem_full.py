# %%
import numpy as np
import scipy.io as sio
# import scipy.misc
# from scipy.misc import imread
# from matplotlib.pyplot import imread
# from cv2 import imread
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"#-----specify device gpu or cpu
from tqdm import tqdm

from watershed_seeds import watershed_seed
from deep_distance_class_estimator import deep_distance_class_estimator
# from patches import patchify, unpatchify

import tensorflow as tf
tf_version = tf.__version__.split('.')
if tf_version[0] == '1':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config = tf_config)
else:
    try:
        tf.config.gpu.set_per_process_memory_growth(True)
    except AttributeError:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# %% set folders
data_path=r'/media/data1/membrane_nucleus_segmentation_classification/Data_and_results';
sub = 'predict-full'
path_weight = os.path.join(data_path, sub, 'weight_mem_full')
path_output = os.path.join(data_path, sub, 'output_mem_full')
n_labels = 3

if not os.path.exists(path_output):
	os.makedirs(path_output)

# %% 
## set parameters for watershed
## footprints relating neighboring pixels
footprint_direct = np.zeros((3,3), dtype = 'bool')
footprint_direct[1,:] = 1
footprint_direct[:,1] = 1
footprint_indirect = np.ones((3,3), dtype = 'bool')
footprint_neighbor = footprint_direct.astype('uint8').copy()
footprint_neighbor[1,1] = 0
footprint_empty = np.array([[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]], dtype='int16')
Lx = Ly = 2048
empty_ext = np.ones((Lx,Ly), dtype='bool')
empty_ext[1:(Lx-1),1:(Ly-1)]=0

## Threshold of EDT. EDT lower than mask_thres are always considered as background
mask_thres=0.2
## Typical radius of a neuron (unit: pixel).
radius = 26
## For two close local maxima, if the ratio of the local minimum between them to the 
## lower local maximum is higher than "dip_ratio", the lower local maximum will be removed from the seeds.
dip_ratio = 0.5
## Maximum distance considered for close local maxima. 
d_peak = radius*2 # 15
## Maximum edge to area ratio of a segmented neuron
e2a_max = 0.2
## Whether further smoothing is applied
further_smooth = True

# %%
## load data
data = sio.loadmat(os.path.join(data_path, 'seg_test_comp_SF.mat'))
membranes = data['channels'][1][0]
shape_full = membranes.shape[:2]
# %%
## process each image
# n = 0

autoencoder = deep_distance_class_estimator(n_labels = n_labels)
# autoencoder.summary()
losses = {'dist': "mean_squared_error",
		'class': "categorical_crossentropy"}
loss_weights = {'dist': 1,
		'class': 1}
autoencoder.compile(loss=losses, loss_weights=loss_weights, optimizer='adam', metrics=['mae','acc'])
weight_file = os.path.join(path_weight, 'weight_EDT_class_{}.h5'.format(4))
autoencoder.load_weights(weight_file)

for n in range(membranes.shape[-1]):
	## load image
	membrane = membranes[:,:,n]
	## Normalize image with respect to image median.
	#membrane=(membrane-np.amin(membrane))*1.0/(np.amax(membrane)-np.amin(membrane))
	#membrane=membrane*1.0/np.median(membrane)
	membrane=(membrane-np.amin(membrane))*1.0/(np.amax(membrane)-np.amin(membrane))
	membrane=membrane*1.0/np.median(membrane)
	SIZE = 256
	# step = 128

	# %%
    ## complie model


	# membrane_patches = patchify(membrane, SIZE, step)  #Step=256 for 256 patches means no overlap
	# print('patched shape:', membrane_patches.shape)
	# membrane_reshape = np.reshape(membrane_patches, (-1,SIZE,SIZE,1))
	# # (predict_data,img_list) = prep_prediction_data(path_predict)
	# EDT_reshape = autoencoder.predict(membrane_reshape, verbose=0)
	# print(EDT_reshape.shape)
	# EDT_patches = np.reshape(EDT_reshape, membrane_patches.shape)
	# # print(type(EDT))
	# EDT_full = unpatchify(EDT_patches, SIZE-step, (1,)+shape_full, 'bilinear')
	# D = EDT_full[0]

	## predict EDT and class
	(dist, classes) = autoencoder.predict(membrane[np.newaxis,:,:,np.newaxis], verbose=0)
	D = dist[0,:,:,0]

	# %%
	## save predicted classification
	img = np.argmax(classes[0],axis=-1).astype('uint8')
	# dir_save = test_name[ind].replace('img','class')
	imsave(os.path.join(path_output, str(n)+'_class.png'), img*127)

	# %% 
	## Use watershed to segment the EDT image
	bw = D>mask_thres
	(labels, local_maxi) = watershed_seed(D, bw, d_peak, dip_ratio, empty_ext, e2a_max, \
		footprint_direct, footprint_indirect, footprint_neighbor, further_smooth)

	## save segmented labels as an image, with enhanced contrast
	labels_max = labels.max()
	labels_contrast = labels + labels_max * labels.astype('bool')
	imsave(os.path.join(path_output, str(n)+'_wseg.png'),labels_contrast.astype('uint16'))
	sio.savemat(os.path.join(path_output,'mem_labels{}.mat'.format(n)),{"masks":labels})

	# %%
	## plot EDT images and watershed seeds 
	fig = plt.figure(figsize=(12, 12))
	plt.imshow(D, cmap=plt.cm.gray, interpolation='nearest')
	plt.autoscale(False)
	plt.plot(local_maxi.nonzero()[1], local_maxi.nonzero()[0], 'r.')
	plt.axis('off')
	plt.title('seeds')
	fig.tight_layout()
	plt.savefig(os.path.join(path_output, str(n)+'_seeds.png'),bbox_inches='tight',format='png',dpi=300)
	# plt.show()
	plt.close(fig)

	# %%
	## plot final segmented neurons
	fig = plt.figure(figsize=(12, 12))
	membrane_clip = np.clip(membrane,0,np.percentile(membrane,99))
	plt.imshow(membrane_clip, cmap=plt.cm.gray, interpolation='nearest')
	plt.autoscale(False)
	plt.contour(labels)
	# for i in tqdm(range(1,labels.max()+1)):
	# 	plt.contour(labels == i)
	# plt.plot(local_maxi.nonzero()[1], local_maxi.nonzero()[0], 'r.')
	plt.axis('off')
	plt.title('masks')
	fig.tight_layout()
	plt.savefig(os.path.join(path_output, str(n)+'_masks.png'),bbox_inches='tight',format='png',dpi=300)
	# plt.show()
	plt.close(fig)

