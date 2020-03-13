function lgraphFinal = initializeFCLayerWeights(net, initializer)

    if isa(net,'SeriesNetwork')
        lgraph = layerGraph(net.Layers);
    else
        lgraph = layerGraph(net);
    end
    layersBefore = lgraph.Layers;
    connectionsBefore = lgraph.Connections;
    idxFC = findCNNFCLayers(net);
    % if the network has more than 2 fc layers, we take the
    % last 2 of those (i.e. fc 7 or fc8)
    if numel(idxFC) > 1
    i1 = idxFC(1);
    i2 = idxFC(end-1);
    i3 = idxFC(end);
    fprintf("Indexes of First FC layer: %d \n",i1);
    fprintf("Indexes of Second FC layer: %d \n",i2);
    fprintf("Indexes of Last FC layer: %d \n",i3);

    %lgraph(i2).WeightsInitializer = "zeros";
    layersBefore(i2).WeightsInitializer = initializer
    layersBefore(i3).WeightsInitializer = initializer
    fprintf("Weights Initializer for fully connected layers %s and %s set to zeros. \n",layersBefore(i2).Name,layersBefore(i3).Name);
    elseif numel(idxFC) == 1
    layersBefore(idxFC).WeightsInitializer = initializer;
    fprintf("Weights Initializer for fully connected layer %d set to zeros. \n",idxFC);
    % if no fully connected layer is found:
    else
    idx = numel(lgraph)-3;
    %layersBefore(idx).WeightsInitializer = initializer;
    fprintf("Weights for fully connected layer equivalent %d (%s) set to zeros. \n",idx,lgraph.Layers(idx).Name);
    end
    fprintf("Creating layer graph....\n")
    layers = layersBefore;
    
    lgraphFinal = createLgraphUsingConnections(layers,connectionsBefore);
end