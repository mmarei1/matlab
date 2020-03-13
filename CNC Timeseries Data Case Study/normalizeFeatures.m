% function to calculate feature mean and standard deviation as vectors
function [featureMean, featureStd] = normalizeFeatures(arr)
    x_v = horzcat(arr{:});
    x_v = cell2mat(x_v);
    featureMean = mean(x_v,2);
    featureStd = std(x_v,0,2);
end