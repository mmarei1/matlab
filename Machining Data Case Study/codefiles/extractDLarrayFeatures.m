% outFeatures = extractDLarrayFeatures(dlnet,hiddenLayers,dataIn,minibatchSize)
% Function to extract dlnetwork features, yielding the feature representation
% to compute transfer loss.
%
% Input arguments:
%   dlnet - the dlnetwork object which contains the feature representation
%   layers, specified by hiddenLayers
%   hiddenLayers - see above
%   dataIn - the data for which the feature representations are computed
%   minibatchSize - the size of the minibatch of data to be computed
% Outputs:
%   outFeatures - a dF x miinibatchSize feature output matrix of type 'double', 
%   where d1 and d2 correspond to the dimensions of the feature representation
%   layer output dimensions. For example, if hiddenLayers has size 1000,
%   and the minibatch has size 16, then outFeatures is a 1000x16 matrix.
% Created by Mohamed Marei, 2020
%%
function outFeatures = extractDLarrayFeatures(dlnet,hiddenLayers,dataIn,minibatchSize)
    din_size = size(cell2mat(table2cell(dataIn(1,'input'))));
    N = minibatchSize;
    din_reshaped = reshape(cell2mat(table2cell(dataIn(1:N,1))),[din_size N]);
    din_dl = dlarray(single(din_reshaped),'SSCB');
    outFeatures = cast(forward(dlnet,din_dl,'Outputs',hiddenLayers(end).Name),'like',0.5);
end