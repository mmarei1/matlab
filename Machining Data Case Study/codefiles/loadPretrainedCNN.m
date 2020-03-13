function [net,modelName] = loadPretrainedCNN(name, nc, lossName, lossVal, idxFrz, classWeights, classBias, meanData)
% load Pre-trained CNN model
%
% [net, modelName] = loadPretrainedCNN(name, nc, lossname, lossval,idxFrz, classWeights, classBias, meanData) loads a
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
%   Additional arguments:
%   -idxFrz: the index before which layers are frozen (i.e. their learning
%   rate is set to zero). Layers after idxFrz will have the default model
%   initial learning rate.
%   -clasWeights: the weights of the output classes
%   -classBias: the class output bias
%   -meanData: the mean data to be used for image normalization.
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
%   Example usage:
% 
%   Load pre-trained model AlexNet, which is a 25-layer CNN trained on
%   a subset of ImageNet (ILSVRC2012). Pass the loss name and loss value
%   arguments to specify a cross-entropy loss function which outputs the 
%   probability of classifying an object into 1 of 8 unique categories, 
%   defined here by the numClasses argument.
% 
% imds = imageDatastore(impath,'LabelSource','none');
% imds.Labels = categorical(strsplit(num2str([1:8])));
% [net, modelname] = loadPretrainedCNN("alexnet",0,"compoundLoss",2);


% Created by Mohamed Marei, 2019.
% -------------------------------------------------------------------------
    
    dateAndTime = datetime('now','Timezone','Local','Format','yyyy-MM-dd');
    exp_name = string(dateAndTime);
    %lossName = varargin(1);
    %lossVal = varargin(end);
    % if loss function not specified, default to mse
    if strcmp(name,'inceptionv3') || strcmp(name,'squeezenet')
        normparamName = 'zerocenter';
    else
        normparamName = 'zerocenter';
    end
    
    if (~strcmp(lossName,"compoundLoss") && numel(nc) <= 0)
        lossName = "mse";
        lossVal = 0;
    end
    
    if numel(nc) > 1
        numClasses = nc;
        lossName = "cross";
    else 
        numClasses = 0;
    end
        [net,bl,assertionResult] = assertAndLoadModel(name, numClasses, lossName, lossVal, classWeights, classBias);
        %fprintf("Final layer name: %s \n",bl(end).Name)
        if isempty(net)
            modelName = NaN;
            fprintf('Model %s not found. \n',name)
            assertionResult
            %fprintf('Assertion result: %s. \n',assertionResult)  
        else
            modelName = strcat(name,"_",exp_name);
            fprintf('Model selected: %s with %d class outputs (0 == regression) \n',name, numel(numClasses))
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
            % initialize normalization with ILSVRC 2012 mean image
            
            [learnableLayer,classLayer, smLayer] = findBottomLayersToReplace(lgraph);
            % save these layers' names into an array of layersToReplace
            %layersToRemove = {learnableLayer.Name,classLayer.Name,smLayer.Name};
            % substitute zero image input layer weights with image mean
            % weights from ILSVRC_2012
            % meanData
            %inputSize = size(lgraph.Layers(1).InputSize;
            if ~isempty(meanData)
                inputLayer = lgraph.Layers(1);    
                sf = inputLayer.InputSize(1)/size(meanData,1);
                %meanData_n = normalize(meanData,'all','range',[0,1]);
                meanData_channels = zeros([1,1,3]);
                meanData_channels(:,:,1) = mean(meanData(:,:,1),'all');
                meanData_channels(:,:,2) = mean(meanData(:,:,2),'all');
                meanData_channels(:,:,3) = mean(meanData(:,:,3),'all');
                %immean = imresize(meanData,[sf sf]);
                immean = meanData_channels;
%                 inputLayer.Normalization = 'zero-center';
%                 inputLayer.Mean = immean;
                inputLayer = imageInputLayer(inputLayer.InputSize,'Name',lgraph.Layers(1).Name,'Normalization',normparamName,'Mean',immean);
                il = lgraph.Layers(1);
                lgraph = replaceLayer(lgraph,il.Name,inputLayer,'ReconnectBy','name');
%                 lgraph = addLayers(lgraph,inputLayer);
%                 lgraph = connectLayers(lgraph,inputLayer.Name,lgraph.Layers(1).Name);
            end
            layersToRemove = {classLayer.Name,smLayer.Name};

            lgraph = removeLayers(lgraph,layersToRemove);
            fl = lgraph.Layers(end);
            lgraph = addLayers(lgraph,bl);
            lgraph = connectLayers(lgraph,fl.Name,bl(1).Name);
            
            %lgraph = addLayers()
            layers = lgraph.Layers;
            connections = lgraph.Connections;
            layers(1:idxFrz) = freezeWeights(layers(1:idxFrz));
            layers(idxFrz+1:end) = unfreezeWeights(layers(idxFrz+1:end),5);
            lgraph = createLgraphUsingConnections(layers,connections);
            net = lgraph;
        end
        
end
