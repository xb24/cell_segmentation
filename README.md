# cell_segmentation

This code used MATLAB and Python. Required Python packages: numpy, scipy, matplotlib, scikit-image, tensorflow, keras, tqdm.

This code used deep distance estimator to estimate the smallest distance of every pixel to background or cell boundary, and then used watershed to segment the distance image. The deep distance estimator was modified from https://github.com/opnumten/single_cell_segmentation, but we did not use the deep cell detector. The same procedure was applied to membrane and nucleus, but the training were performed separately. Then we compared the segmented membrane and nucleus masks and found matched neurons. The membrane CNN had another output, which classified each pixel as one of the three classes: background, cell body, or membrane. We then assigned each membrane pixel to the closest segmented neuron.

