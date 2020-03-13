function indexes = findCNNMaxPoolingLayers(net)
isPoolingLayers = arrayfun(@(l)...
    (isa(l,'nnet.cnn.layer.MaxPooling2DLayer')),...
    net.Layers);
tf = isPoolingLayers;
indexes = find(tf);
end