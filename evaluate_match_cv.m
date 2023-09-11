% Evaluate the segmentation accuracy using Recall, Precision, and F1
eval_mem_cv; % membrane masks
eval_nuc_cv; % nucleus masks
% Summarize the accuracy of membrane classification using IoU
summarize_mem_class_cv; % membrane classes
% Quickly display the accuracy metrics in the future.
% disp_mem_mask_class_cv; 

% Show the cropped membrane and nucleus images and segmented masks
plot_output_mem_cv; % membrane images and masks
plot_output_nuc_cv; % nucleus images and masks
plot_output_mem_class_cv; % membrane images and classes

% Match segmented neuron masks and membrane pixels
match_cell_mem_cv;

% Match the segmented masks between paired membrane and nucleus images.
plot_output_match_cv;

%%
% exit;