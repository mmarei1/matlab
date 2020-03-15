%% transferLossLayer 
% custom regression layer with mean-squared-error loss and a pre-assumed mmd value.
%% Syntax
% layer = transferLossLayer(name, lossType, netArray, sourceData,
% ... targetData, hiddenLayers, weights) 
%% Input Arguments:
%   Name - layer name
%   LossType:  loss type for training; can either be:
%       "mse-mmd": normalized MSE-MMD loss
%       "mse": Mean-squared error loss
%   layerGraph: a layerGraph containing the preceding network layers
%   sourceData: the source domain dataset
%   targetData: the target domain dataset
%   hiddenLayers: a layer array specifying the indexes of the layers to
%   compute the feature representations (or embeddings)
%   weights: a 1x2 weight vector where sum(w1+w2) = 1
%   If the weight vector is [1,0], the layer loss function changes to MSE
%   by default.
%%   Created by Mohamed Marei, 2020
%%
classdef transferLossLayerD2 < nnet.layer.RegressionLayer
    properties
        Weights
        LossFunction
        LayerGraph
        FeatureOutputs
        SourceData;
        TargetData;
        
    end
    
    methods
        function layer = transferLossLayerD2(name, lossType, netArray, sourceData, targetData,hiddenLayers, weights)
            % layer = transferLossLayer(name) creates a
            % mean-squared-error + normalized MKMMD regression layer  and specifies the layer
            % name, in addition to the variables used to calculate the loss type.
			
            % Set layer name.
            layer.Name = name;
            layer.Type = 'RegressionOutputLayer';
            layer.LayerGraph = netArray;
            % Set layer description.
            if strcmp(lossType,"mse-mmd")
                layer.Description = 'Mean Squared Error with MK-MMD loss';
                layer.Weights = weights;            
                layer.FeatureOutputs = netArray.Layers(hiddenLayers');
                layer.SourceData = readall(sourceData);
                layer.TargetData = readall(targetData);
                layer.LossFunction = "mse-mmd";
            else
                layer.Description  = "Mean Squared Error";
                layer.LossFunction = "mse";
                layer.Weights = [1, 0];
                layer.FeatureOutputs = {};
                layer.SourceData = [];
                layer.TargetData = [];
            end
        end
        
%         function Z = predict(layer, varargin)
%             % run the prediction function on the inputs
%             X = varargin;
%             wmse = layer.Weights(1);
%             wnmmd = layer.Weights(2);
%             mmd = layer.NMMD;
%             X1 = X(1);
%             sz = size(X1);
%             Z = zeros(sz,'like',X1);
%             
%             for i = 1:layer.NumInputs
%                 Z = Z + wmse.*X1 + wnmmd.*mmd ;
%             end
%             
%         end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MSE loss between
            % the predictions Y and the training targets T.
            wmse = layer.Weights(1);
            wnmmd = layer.Weights(2);
            R = size(Y,3);          % number of responses
            N = size(Y,4);          % number of observations
            % Extract source and target data from the layer
            
            % Calculate MAE over number of outputs.
            meanSquaredError = sum(((Y-T).^2),3)/R;
            
            % Output MMD: compute the value of MMD and normalize
            % Take mean over mini-batch.
            % ONLY compute MMD weights if wmmd > 0
            if wnmmd > 0
                sourceData = layer.SourceData(1:N,1);
                targetData = layer.TargetData(1:N,1);
                % define dlnet in this step
                dlnet = dlnetwork(layer.LayerGraph);
                hiddenLayers = layer.FeatureOutputs;
                % extract features here
                sourceFeatures = extractDLarrayFeatures(dlnet,hiddenLayers,sourceData,N);
                targetFeatures = extractDLarrayFeatures(dlnet,hiddenLayers,targetData,N);
                [mmd,~] = mmdLossD2(sourceFeatures,targetFeatures);
            else
                mmd = 0;
            end
            loss = wmse .* sum(meanSquaredError)/N + wnmmd .* mmd;
        end
%       
%       Compute Backward Loss of MSE-MMD
        function dLdY = backwardLoss(layer, Y, T)
            % alpha = 0.005;
            % Prepare weights
            wmse = layer.Weights(1);
            wnmmd = layer.Weights(2);
            R = size(Y,3);
            N = size(Y,4);
            % compute the MSE inverse gradient by multiplying it by double
            % the MSE and the MSE weight
            dLdMSE = 2*wmse*N*(Y-T);
            if wnmmd > 0
                sourceData = layer.SourceData(1:N,1);
                targetData = layer.TargetData(1:N,1);
                % define dlnet in this step
                dlnet = dlnetwork(layer.LayerGraph);
                hiddenLayers = layer.FeatureOutputs;
                % extract features here
                sourceFeatures = extractDLarrayFeatures(dlnet,hiddenLayers,sourceData,N);
                targetFeatures = extractDLarrayFeatures(dlnet,hiddenLayers,targetData,N);
                [~,gradMMD] = mmdLossD2(sourceFeatures,targetFeatures,N,dlnet,hiddenLayers);
            %gradMMD = gradMMD*wnmmd;
            else
                gradMMD = 0;
            end
            % weight the MMD loss by its weight during back-prop
            dLdMMD = wnmmd*gradMMD;
            
            dLdY = dLdMSE + dLdMMD;
        end
    end
end