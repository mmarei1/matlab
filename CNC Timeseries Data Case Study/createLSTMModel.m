%% Function to create LSTM model from inputs
% [larrays, ldetails] = createLSTMModel(initializer, inputFeatDim, width)
% creates a Long Short-Term Memory (LSTM) model with the following
% structure:
%      Sequence Input Layer of size (inputFeatDim-by-s) where s is the
%      input data sequence length
%      --Connected to:
%      4x[LSTM Layers connected to 4 Fully Connected (FC) Layers with the following
%         parameters:
%                    HiddenUnits = inputFeatDim*10*width;
%         FC Layers with N/2 HiddenUnits                                       ]
%        FullyConnectedLayer with 1 output (numResponses)
%        RegressionLayer with RMSE loss
%
% Created by Mohamed Marei, 2020
%%
function [larrays, ldetails] = createLSTMModel(initializer, inputFeatureDimension, widthFactor)

    if ~ ( strcmp(initializer,"Glorot") ||  strcmp(initializer,"He") || strcmp(initializer,"narrow-normal"))
        warning("Initializer must be either Glorot, He or narrow-normal")
        initializer = "";
        ldetails = "";
        larrays = cell(1);
        
    else
        numResponses = 1;
        numFeatures = inputFeatureDimension;
        numHiddenUnits = 8*numFeatures;
        ldetails = sprintf("LSTM with 4 hidden-fc layers with %d hidden units in its first LSTM layer, with %s weights initializer",numHiddenUnits*widthFactor,initializer)
        larrays = [... 
            sequenceInputLayer(numFeatures,'Name','InputLayer1'),...
            lstmLayer(numHiddenUnits*widthFactor,'OutputMode','sequence','InputWeightsInitializer',initializer,'Name','LSTM_Layer1'),...
            fullyConnectedLayer(numHiddenUnits*widthFactor/2,'Name','FC_Layer1'),...
            lstmLayer(numHiddenUnits*widthFactor/2,'OutputMode','sequence','InputWeightsInitializer',initializer,'Name','LSTM_Layer2'),... 
            fullyConnectedLayer(numHiddenUnits*widthFactor/4,'Name','FC_Layer2'),...
            lstmLayer(numHiddenUnits*widthFactor/4,'OutputMode','sequence','InputWeightsInitializer',initializer,'Name','LSTM_Layer3'),... 
            fullyConnectedLayer(numHiddenUnits,'Name','FC_Layer3'),...
            lstmLayer(numHiddenUnits/2,'OutputMode','sequence','InputWeightsInitializer',initializer','Name','LSTM_Layer4'),... 
            fullyConnectedLayer(numResponses,'name','FC_Layer_FINAL'),...
            regressionLayer('Name','RegressionLayer'),...
            ];
    end
end