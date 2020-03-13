function finalLayers = createMMD_CNN(layers_in,lossType,mmd_weights,sourceData,targetData)
    % step 1: remove output layers
    if isa(layers_in,'SeriesNetwork')
        lgraph = layerGraph(layers_in)
    else
        lgraph = layerGraph(layers_in.Layers);
    end
    [~,sml,ol] = findBottomLayersToReplace(lgraph);
    layersToRemove = {sml.Name,ol.Name};
    
    % Remove unwanted output layers
    lgraph = removeLayers(lgraph,layersToRemove);
    hiddenLayers = findCNNFCLayers(lgraph);
    % determine if number of hidden layers will be 1 or 2 based on present
    % hidden layers in the CNN
    
    if numel(hiddenLayers)>=2
        hiddenLayers = hiddenLayers([2,3])
    else
        hiddenLayers = hiddenLayers(end);
    end
    % add required input layers
    newLayers = [...
            batchNormalizationLayer('Name','bn_final'),...
            reluLayer('Name','relu_head_1'),...
            fullyConnectedLayer(256,'Name','fc_downsampling_2','BiasLearnRateFactor',5),...
            batchNormalizationLayer('Name','bn2_final'),...
            reluLayer('Name','relu_head_2'),...
            fullyConnectedLayer(128,'Name','fc_downsampling_3','BiasLearnRateFactor',5),...
            batchNormalizationLayer('Name','bn3_final'),...
            sigmoidActivationLayer(1,'siglayer'),...
            fullyConnectedLayer(1,'Name','fcfinal','BiasLearnRateFactor',2),...
        ];
    lastLayer = lgraph.Layers(end);
    lgraph = addLayers(lgraph,newLayers);
    lgraph = connectLayers(lgraph,lastLayer.Name,newLayers(1).Name);
    
    
    % finally, add new output layer
    if strcmp(lossType,"mse-mmd")
        fprintf("CNN Training loss: %s loss \n",lossType)
        layerOut = transferLossLayer("mse_mmd_Reg","mse-mmd",...
            lgraph,sourceData,targetData,hiddenLayers,mmd_weights)    
    else
        fprintf("CNN Training loss: MSE loss \n")
        layerOut = regressionLayer('name','mse_reg'); 
    end
    lastLayer = lgraph.Layers(end);
    lgraph = addLayers(lgraph,layerOut);
    lgraph = connectLayers(lgraph,lastLayer.Name,layerOut.Name);
    
    idxFrz = 12;
    allLayers = lgraph.Layers;
    
    allLayers(1:idxFrz) = freezeWeights(allLayers(1:idxFrz));
    allConnections = lgraph.Connections;
    % create layer graph using connections
    finalLayers = createLgraphUsingConnections(allLayers,allConnections)
    analyzeNetwork(finalLayers)
end