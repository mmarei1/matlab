% 
%%lgraph = cnn_lstm_prototype(cnn,depth,width,initializer)
%
% Function to add fc layers to an input CNN.
% Create convolutional LSTM model with specified number of LSTM layers and
% weight-bias initializer.
% 
% Input arguments:
%   cnn: base CNN model name
%   depth: LSTM-Fully Connected layer array depth (>1)
%   width: Number of hidden units in the first LSTM layer, cascading down
%   by half for each subsequent LSTM layer. Each fully connected layer
%   following the LSTM layer will have half the outputs of the preceding
%   width
%   initializer: Input weights initializer for the LSTM layers. Default is
%   'Glorot'
%
%   Example usage:
%
%   Create a CNN-LSTM based on AlexNet.
%   
%   alexnet_lstm = cnn_lstm_prototype('alexnet',2,128,'Glorot')
%
% Created by Mohamed Marei, 2020
% 
function lgraph = cnn_lstm_prototype(cnn_name,depth,width,initializer)
    %depth = 3;
    %width = 128;
    %cnn = resnet18;
    try
        cnn = eval(cnn_name);
        fprintf('Base CNN: %s\n',cnn_name);
    catch
        msgID = 'cnn_lstm_prototype:NoPretrainedCNN';
        msg = sprintf('Unable to find existing CNN %s\n',cnn_name);
        baseException = MException(msgID,msg);
        throw(baseException)
    end
    
    if isa(cnn,'SeriesNetwork')
        lgraph = layerGraph(cnn.Layers)
    else
        lgraph = layerGraph(cnn)
    end
    
    inSize = cnn.Layers(1).InputSize;
    inName = lgraph.Layers(1).Name;
    % Step 1: create an input layer array comprising:
    layersin = [sequenceInputLayer(inSize,'Name','seq_in'),sequenceFoldingLayer('Name',inName)]
    
    lgraph = replaceLayer(lgraph,inName,layersin,'ReconnectBy','Name')

    % retain the existing FC-1000
    layersconnect = [...
        sequenceUnfoldingLayer('Name','fc2seqUn'),...
        flattenLayer('Name','flatten'),...
    ]
    lstm_fc = [];
    % repeat lstm-fc layers with decreasing depth
    for i = 1:depth-1
        nwidth_lstm = width/(2^(i-1));
        nwidth_fc = width/(2^(i));
        strname_lstm = ["lstm_"+num2str(nwidth_lstm)];
        strname_fc = ["fc_"+num2str(nwidth_fc)];
        tmplayers = [...
            lstmLayer(nwidth_lstm,'Name',strname_lstm,'InputWeightsInitializer',initializer),...
            fullyConnectedLayer(nwidth_fc,'Name',strname_fc,'BiasLearnRateFactor',5),...
            ];
        lstm_fc = [lstm_fc,tmplayers]
    end
    % apply the sigmoidal activation function to last outputs
    final_nhu = lstm_fc(end).OutputSize;
    strname_lstm = ["lstm_"+num2str(final_nhu)];
    fc_out = [lstmLayer(final_nhu,'Name',strname_lstm),...
        fullyConnectedLayer(1,'Name','fc_final','BiasLearnRateFactor',2),...
        sigmoidActivationLayer(1,'sigmoid_final'),...
        regressionLayer('Name','regout')]
    % add the layersconnect to the top of the end layer array stack
    layersconnect = [layersconnect,lstm_fc,fc_out]
    [~,smL,classL] = findBottomLayersToReplace(lgraph);
    layersToRemove = {smL.Name,classL.Name};
    lgraph = removeLayers(lgraph,layersToRemove);
    % get the name of the new FC layer
    fcname = lgraph.Layers(end).Name
    lgraph = addLayers(lgraph,layersconnect);
    sf_mb = strcat(layersin(2).Name ,"/miniBatchSize")
    su_mb = strcat(layersconnect(1).Name,"/miniBatchSize")
    su_in = strcat(layersconnect(1).Name,"/in")
    % connect output of fully connected layer to 
    lgraph = connectLayers(lgraph,sf_mb,su_mb);
    % connect by minibatch
    lgraph = connectLayers(lgraph,fcname,su_in)

    %analyzeNetwork(lgraph)
end
