% Show the full membrane and nucleus images and segmented masks
plot_output_mem_class_full; % membrane images and classes

% Match segmented neuron masks and membrane pixels
match_cell_mem_full;

% Match the segmented masks between paired membrane and nucleus images.
plot_output_match_full;

%%
% exit;