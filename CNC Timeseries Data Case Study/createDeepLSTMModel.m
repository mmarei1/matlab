%% Function to create LSTM model from inputs
% [larrays, ldetails] = createLSTMModel(initializer, depth, inputFeatDim, width, plotflag)
% creates a Long Short-Term Memory (LSTM) model with the following
% structure:
%      Sequence Input Layer of size (inputFeatDim-by-s) where s is the
%      input data sequence length
%      --Connected to:
%      depthx[LSTM Layers connected to Fully Connected (FC) Layers with the following
%         parameters:
%                    HiddenUnits = inputFeatDim*10*width;
%        FC Layers with N/2 HiddenUnits                                       ]
%        FullyConnectedLayer with 1 output (numResponses)
%        RegressionLayer with RMSE loss
%     Additionally, plot the layer graph created if plotflag is true.
%
% Created by Mohamed Marei, 2020
%%
function [larrays, ldetails] = createDeepLSTMModel(initializer, depth, inputFeatureDimension, widthFactor,plotflag)

    if ~ ( strcmp(initializer,"Glorot") ||  strcmp(initializer,"He") || strcmp(initializer,"narrow-normal"))
        warning("Initializer must be either Glorot, He or narrow-normal")
        initializer = "";
        ldetails = "";
        larrays = cell(1);
         
    else
        numResponses = 1;
        numFeatures = inputFeatureDimension;
        numHiddenUnits = 8*numFeatures;
        ldetails = sprintf("LSTM Structure: %d hidden-fc layer pairs with %d hidden units in its first LSTM layer, %s input weights initializer",depth,numHiddenUnits*widthFactor,initializer)
        
        lstm_fc =[sequenceInputLayer(numFeatures,'Name','InputLayer'),...
                  ];
        for i=1:depth-1
            str_l = ["LSTM_Layer_"+num2str(i)];
            fc_l = ["FC_Layer_"+num2str(i)];
            % create LSTM with twice as many hidden units as fully
            % connected outputs
            nhu = numHiddenUnits*widthFactor/(2^(i-1));
            nfcOut = numHiddenUnits*widthFactor/(2^i);
            tmpArr = [...
                lstmLayer(nhu,'OutputMode','sequence','InputWeightsInitializer',initializer,'Name',str_l),...
                fullyConnectedLayer(nfcOut,'Name',fc_l),...
            ];
            lstm_fc = [lstm_fc,...
                tmpArr];
        end
        str_l = ["LSTM_Layer_"+num2str(depth)];
        fc_l = ["FC_Layer_"+num2str(depth)];
        larrays = [... 
            lstm_fc,...
            lstmLayer(numHiddenUnits*widthFactor/2^(depth-1),'OutputMode','sequence','InputWeightsInitializer',initializer','Name',str_l),... 
            fullyConnectedLayer(numResponses,'name',fc_l),...
            regressionLayer('Name','RegressionLayer'),...
            ];
    end
    
    if plotflag == true
    %title(ldetails)
    plotLayerGraph(layerGraph(larrays),ldetails)
    end
end