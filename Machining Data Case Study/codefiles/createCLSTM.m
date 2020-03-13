%% createCLSTM
% create convolutional LSTM model with specified number of LSTM layers and
% weight-bias initializer.
% [lstm_cnn, details ] = createCLSTM(convnet, lstmDesnity, depth, initializer)
%
% Mohamed Marei, 2020
%%
function [lstm_cnn, details] = createCLSTM(convnet,numHiddenUnits,depth,initializer)
    clear lgraph tempLayers;
    
    if isempty(initializer)
        initializer = 'Glorot';
        fprintf('Setting default initializer to %s.\n',initializer)
    end
    imInputSize = convnet.Layers(1).InputSize;
        if isempty(numHiddenUnits)
           numHiddenUnits = 256;
           fprintf("Using default LSTM density of %d \n", numHiddenUnits);
        end
        lgraph = layerGraph(convnet.Layers);
%         if isa(lgraph.Layers(end),'nnet.cnn.layer.ClassificationOutputLayer')
%             try
%             [fclayer,classLayer,smLayer] = findBottomLayersToReplace(lgraph);
%             catch me
%                 warning("Can't find layers to remove - no output layers found");
%                 % assume softmax layer is penultimate layer of the network
%                 smLayer = lgraph.Layers(end-1);
%                 classLayer = lgraph.Layers(end);
%                 all_fc = findCNNFCLayers(lgraph);
%                 fclayer = lgraph.Layers(all_fc(end));
%                 layersToRemove = {classLayer.Name,smLayer.Name};
%             end
%         else
%             fclayer = lgraph.Layers(end-1);
%             layersToRemove = {lgraph.Layers(end).Name};
%         end
        [fclayer,classL,smL] = findBottomLayersToReplace(lgraph);
        layersToRemove = {classL.Name,smL.Name};
        %fclayer = findCNNFCLayers(lgraph);
        
        lgraph = removeLayers(lgraph,lgraph.Layers(1).Name);
        tempLayers1 = ...
            [sequenceInputLayer(imInputSize,'Name','SequenceInputLayer','Normalization','zerocenter'),...
            sequenceFoldingLayer('Name','seqfold'),...
            ];   
        lgraph = removeLayers(lgraph,layersToRemove);
        lgraph = addLayers(lgraph,tempLayers1);
        %%
        %tempLayers = [];
        unfoldingFlatten = ...
            [sequenceUnfoldingLayer('Name','sequnfold'),...
            flattenLayer('Name','flattenLayer')];
        lstm_fc = [];
        for i=1:depth-1
            str_l = ["LSTM_Layer_"+num2str(i)];
            fc_l = ["FC_Layer_"+num2str(i)];
            % create LSTM with twice as many hidden units as fully
            % connected outputs
            nhu = numHiddenUnits/(2^(i-1));
            nfcOut = numHiddenUnits/(2^i);
            tmpArr = [...
                lstmLayer(nhu,'OutputMode','sequence','InputWeightsInitializer',initializer,'Name',str_l),...
                fullyConnectedLayer(nfcOut,'Name',fc_l),...
            ];
            lstm_fc = [lstm_fc,...
                tmpArr];
        end
        outLayers = [...
            sigmoidActivationLayer(1,'Sigmoid_final'),...
            regressionLayer('Name','RegressionOut')
        ];
        tempLayers2 = [...
            unfoldingFlatten,...
            lstm_fc,...
            outLayers
            ];
        lgraph = addLayers(lgraph,tempLayers2);
    % connect everything up
        lgraph = connectLayers(lgraph,"seqfold/out","conv1");
        lgraph = connectLayers(lgraph,"seqfold/miniBatchSize","sequnfold/miniBatchSize");
        lgraph = connectLayers(lgraph,fclayer.Name,"sequnfold/in");
        lstm_cnn = lgraph;
        details = sprintf("CNN-LSTM with %d LSTM layers initialized via %s - hidden units of first layer: %d", depth, initializer, numHiddenUnits);
end