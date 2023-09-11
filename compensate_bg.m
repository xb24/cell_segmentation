clear;
data_path = ['/media/data1/membrane_nucleus_segmentation_classification'];
load(fullfile(data_path,'seg_test.mat'))
leng = size(channels{1},3);
channels_comp = channels;

%% Apply background compensation to the membrane and nucleus images
for i=1:2
    temp = channels{i};
    temp_comp = temp;
    for j = 1:leng
        fprintf(num2str(j));
        temp_comp(:,:,j) = imnormalize2_q(temp(:,:,j),400,0.1,4);
    end
    channels_comp{i} = temp_comp;
end

channels = channels_comp;
save(fullfile(data_path,'seg_test_comp.mat'),'channels')
