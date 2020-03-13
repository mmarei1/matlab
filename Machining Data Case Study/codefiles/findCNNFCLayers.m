% utility function to extract the fc layers
function indexes = findCNNFCLayers(net)
isFCLayer = arrayfun(@(l)...
    (isa(l,'nnet.cnn.layer.FullyConnectedLayer')),...
    net.Layers);
tf = isFCLayer;
indexes = find(tf);
end