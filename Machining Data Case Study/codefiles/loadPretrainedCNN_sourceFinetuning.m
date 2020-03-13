function [net,modelName] = loadPretrainedCNN(name,nc,lossName, lossVal)
% load Pre-trained CNN model
%
% [net, modelName] = loadPretrainedCNN(name, nc, lossname, lossval) loads a
% user-specified pre-trained CNN with nc classification outputs. If nc == 0
% then the model is returned with a regression output layer. Specify
% optional arguments "lossname" and "lossVal" to change the model loss
% function to one of two options:
%
%  - LossName = "mse" - the default Mean Square Error function. 
%    Specify the lossval argument as 0.
%  - LossName = "compoundLoss" - the loss function is a weighted Mean Square 
%  Error loss with Maximum Mean Discrepancy term, whose value (assumed to be a 
%  constant) is fixed as "lossval".
% 
%   Example usage:
% 
%   Load pre-trained model AlexNet, which is a 25-layer CNN trained on
%   a subset of ImageNet (ILSVRC2012). Pass the loss name and loss value
%   arguments to specify a Weighted MSE-MMD loss function which assumes the
%   MMD value between source and target data to be lossVal (2 here).
% 
% [net, modelname] = loadPretrainedCNN("alexnet",0,"compoundLoss",2);
% 
% Created by Mohamed Marei, 2019.
% -------------------------------------------------------------------------
    
    dateAndTime = datetime('now','Timezone','Local','Format','yyyy-MM-dd');
    exp_name = string(dateAndTime);
    %lossName = varargin(1);
    %lossVal = varargin(end);
    % if loss function not specified, default to mse
    if (~strcmp(lossName,"compoundLoss") && numel(nc) <= 0)
        lossName = "mse";
        lossVal = 0;
    end
    
%     
%     resultsStruct = struct(...
%     'NetworkName',{modelName},...
% 	'NetworkTrainingVariant',{optimizer},...
% 	'NetworkSize',{0},...
% 	'TrainingOptions',{},...
% 	'Performance',{0},...
% 	'dateAndTime',{dateAndTime},...,
%     'Summary',{"Summary goes here"});
% switch case through models available
% if the length of variable arguments exceeds 1, then the model is a
% classification model with numClasses = vararg(2)
    if numel(nc) > 1
        numClasses = nc;
        lossName = "cross"
    else 
        numClasses = 0;
    end
        [net,bl] = assertAndLoadModel(name, numClasses, lossName, lossVal);
        fprintf("Final layer name: %s \n",bl(end).Name)
        if isempty(net)
            modelName = NaN;
            %fprintf('Model %s not found. \n',name)        
        else
            modelName = strcat(name,"_",exp_name);
            fprintf('Model selected: %s with %d class outputs (0 == regression) \n',name, numel(numClasses)-1)
            fprintf('Loss function: %s \n',lossName)
            % handle network if SeriesNetwork or DAG
            if isa(net,'SeriesNetwork')
                lgraph = layerGraph(net.Layers);
            else
                lgraph = layerGraph(net);
            end
            
            % Set initializers for last FC layers to zero
            try
            layersInitialized = initializeFCLayerWeights(net,"zeros");
            catch 
                warning("Model does not have a fully connected layer!");
                layersInitialized = lgraph;
            end
            lgraph = layersInitialized;
            
            % TODO: create a function here to reuse for "hot-swapping"
            [learnableLayer,classLayer, smLayer] = findBottomLayersToReplace(lgraph);
            % save these layers' names into an array of layersToReplace
            %layersToRemove = {learnableLayer.Name,classLayer.Name,smLayer.Name};
            layersToRemove = {classLayer.Name,smLayer.Name};
            
            %lgraph = layersInitialized;
                        
            lgraph = removeLayers(lgraph,layersToRemove);
            
            layers_before = lgraph.Layers;
            connections_before = lgraph.Connections;

            % Freeze the top 10 layers
            idxFrz = 10;
            layers_before(1:idxFrz) = freezeWeights(layers_before(1:idxFrz));
            layers_before(idxFrz+1:end) = unfreezeWeights(layers_before(idxFrz+1:end),20);
            
%            idxFC = findCNNFCLayers(net)
            % if the network has more than 2 fc layers, we take the
            % last 2 of those (i.e. fc 7 or fc8)
%             if numel(idxFC) > 1
%                 i1 = idxFC(1);
%                 i2 = idxFC(end-1);
%                 i3 = idxFC(end);
%                 fprintf("Indexes of First FC layer: %d \n",i1);
%                 fprintf("Indexes of Second FC layer: %d \n",i2);
%                 fprintf("Indexes of Last FC layer: %d \n",i3);
% 
%                 %lgraph(i2).WeightsInitializer = "zeros";
%                 layersBefore(i2).WeightsInitializer = "zeros"
%                 layersBefore(i3).WeightsInitializer = "zeros"
%                 fprintf("Weights Initializer for fully connected layers %s and %s set to zeros. \n",lgraph.Layers(i2).Name,lgraph(i3).Name);
%             elseif numel(idxFC) == 1
%                 layersBefore(idxFC).WeightsInitializer = "zeros";
%                 fprintf("Weights Initializer for fully connected layer %d set to zeros. \n",idxFC);
%             else
%                 idx = numel(lgraph)-3;
%                 layersBefore(idx).WeightsInitializer = "zeros";
%                 fprintf("Weights Initializer for fully connected layer equivalent %d set to zeros. \n",idx);
%             end
            
            fl = lgraph.Layers(end);
            prefinalLayer = lgraph.Layers(end).Name;
            lgraph = addLayers(lgraph,bl);
            
            lgraph = connectLayers(lgraph,prefinalLayer,bl(1).Name);
            layers = lgraph.Layers;
            connections = lgraph.Connections;
            lgraph = createLgraphUsingConnections(layers,connections);
            
            net = lgraph;
            % TODO: End new function here
        end
        
end
