# %%
import numpy as np
# import scipy.misc
# from scipy.misc import imread
# from matplotlib.pyplot import imread
# from cv2 import imread
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
import os
from os import listdir

from watershed_seeds import watershed_seed

# %% set folders
data_path=r'/media/data1/membrane_nucleus_segmentation_classification/Data_and_results'
sub = 'cross-validation'
path_EDT = os.path.join(data_path, sub, 'EDT_mem_cv')
dist_list = sorted(listdir(path_EDT))
path_output = os.path.join(data_path, sub, 'output_mem_cv')
if not os.path.exists(path_output):
	os.makedirs(path_output)


# %% set groups for 4-fold cross-validation
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

# %% set parameters for watershed
## footprints relating neighboring pixels
footprint_direct = np.zeros((3,3), dtype = 'bool')
footprint_direct[1,:] = 1
footprint_direct[:,1] = 1
footprint_indirect = np.ones((3,3), dtype = 'bool')
footprint_neighbor = footprint_direct.astype('uint8').copy()
footprint_neighbor[1,1] = 0
footprint_empty = np.array([[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]], dtype='int16')
Lx = Ly = 256
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


# %% for each round of cross-validation
for cv in range(4): 
	print('Cross-validation round',cv)
	testset = group[cv]
	dir_test_dist = [dist_list[x] for x in testset]

	# %% for each image
	for (i,test_dist) in enumerate(dir_test_dist):
		## load EDT predicted by CNN
		img=imread(path_EDT+'/'+test_dist)
		img_h=img.shape[0]
		img_w=img.shape[1]
		D=np.reshape(img,(img_h,img_w))

		## Use watershed to segment the EDT image
		bw = D>mask_thres
		(labels, local_maxi) = watershed_seed(D, bw, d_peak, dip_ratio, empty_ext, e2a_max, \
			footprint_direct, footprint_indirect, footprint_neighbor, further_smooth)

		## save segmented labels as an image, with enhanced contrast
		labels_max = labels.max()
		labels_contrast = labels + labels_max * labels.astype('bool')
		imsave(os.path.join(path_output, test_dist[0:len(test_dist)-4]+'_wseg.png'),labels_contrast)
		
		# %% 
		## plot EDT images, watershed seeds, and final segmented neurons
		fig, axes = plt.subplots(ncols=2, figsize=(6, 3), sharex=True, sharey=True,
								subplot_kw={'adjustable': 'box'}) # -forced
		ax = axes.ravel()

		ax[0].imshow(img, cmap=plt.cm.gray)
		ax[0].autoscale(False)
		ax[0].plot(local_maxi.nonzero()[1], local_maxi.nonzero()[0], 'r.')
		ax[0].axis('off')
		ax[0].set_title('seeds')

		ax[1].imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
		ax[1].set_title('Separated objects')

		for a in ax:
			a.set_axis_off()
		fig.tight_layout()
		
		plt.savefig(os.path.join(path_output, test_dist[0:len(test_dist)-4]+'_ws_seg.png'),bbox_inches='tight',format='png',dpi=300)
		# plt.show()
		plt.close(fig)

