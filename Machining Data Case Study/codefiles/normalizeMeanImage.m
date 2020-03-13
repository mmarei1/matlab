function [meanIm, channels_mean, channels_std] = normalizeMeanImage(mean_data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
dim = size(mean_data,1).^2;
meanIm = zeros(size(mean_data));
channels_mean = [0,0,0];
channels_std = [0,0,0];
for i=1:3
    mv = reshape(mean_data(:,:,i),[dim, 1]);
    channels_mean(i) = mean(mv);
    channels_std(i) = std(mv);
    meanIm(:,:,i) = (mean_data(:,:,i) - channels_mean(i))/channels_std(i);
end

end

