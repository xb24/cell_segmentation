%% Pre-processing
% Correct the manual labels
% correct2;
% Compensate for nonuniform background intensity
compensate_bg;
% Apply high-pass filter to membrane images to enhance the membranes
high_pass_filter;

%% Move data not used in future to another folder to avoid mistaken use.
% dir_unused = fullfile(data_path,'unused');
% if ~exist(dir_unused,'dir')
%     mkdir(dir_unused);
% end
% movefile(fullfile(data_path,'seg_test.mat'), ...
%     fullfile(dir_unused,'seg_test.mat'))
% movefile(fullfile(data_path,'seg_test_comp.mat'), ...
%     fullfile(dir_unused,'seg_test_comp.mat'))
% movefile(fullfile(data_path,'mask3.mat'), ...
%     fullfile(dir_unused,'mask3.mat'))

%% Crop to patches
% Crop the membrane and nucleus images to 256 * 256 patches
% crop_mem; % membrane masks and distance
% crop_mem_edge; % membrane pixels
% crop_nuc; % nucleus masks and distance

% Show the cropped membrane and nucleus images and manual masks
% plot_GT_mem; % membrane images and masks
% plot_GT_nuc; % nucleus images and masks

%%
% exit;