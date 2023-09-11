clear;
data_path = '/media/data1/membrane_nucleus_segmentation_classification';
load(fullfile(data_path,'seg_test_comp.mat'))
leng = size(channels{1},3);
channels_HPF = channels;

%% Spatial high-pass filter
sigmafilt = 3; %tune from 1 to 5 (decimals are ok)
sizefilt = 16;
[xfilt, yfilt] = meshgrid(-sizefilt:sizefilt,-sizefilt:sizefilt);
kernel2 = exp(-(xfilt.^2+yfilt.^2)/2/sigmafilt^2);
kernel2 = kernel2/sum(kernel2(:));
kernel3 = zeros(size(xfilt));
kernel3(sizefilt+1,sizefilt+1) = 1;
kernel3 = kernel3-kernel2;

%% Apply spatial high-pass filtering to the membrane images but not nucleus images
for i=2 % 1:2
    temp = single(channels{i});
    temp_HPF = temp;
    for j = 1:leng
        fprintf(num2str(j));
        temp_HPF(:,:,j) = single(imfilter(double(temp(:,:,j)),kernel3,'symmetric','same'));
    end
    channels_HPF{i} = temp_HPF;
end

channels = channels_HPF;
save(fullfile(data_path,'seg_test_comp_SF.mat'),'channels')
