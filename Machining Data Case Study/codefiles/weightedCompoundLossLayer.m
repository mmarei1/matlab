classdef weightedCompoundLossLayer < nnet.layer.RegressionLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        LossWeights
        MMD_Metric
    end

    methods
        function layer = weightedCompoundLossLayer(MMD_Metric, name)
            % layer = compoundLossLayer(MMD_Metric, name) creates a
            % weighted MK-MMD MSE loss layer. MMD_Metric is a Maximum Mean
            % Discrepancy value equating to the difference between the
            % source and target domain distributions of the pre-trained CNN
            % data.
            % 
            % layer =compoundLossLayer(MMD_Metric, name)
            % additionally specifies the layer name. 

            % Set loss weights
            r = rand(1);
            layer.LossWeights = sort([r,1-r],'descend');
            layer.MMD_Metric = MMD_Metric;
            % Set layer name
            if nargin == 2
                layer.Name = name;
            end

            % Set layer description
            layer.Description = 'Regression output weighted summation of the MMD and MSE loss';
            
            layer.ResponseNames = '';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted MMD-NMSE
            % loss for the training predictions and targets

            wmse = layer.LossWeights(1);
            wnmmd = layer.LossWeights(2);
            mmd = layer.MMD_Metric;
            % Calculate MAE.
            % divide by number of responses (=1)
            R = size(Y,3);
            meanSquaredError = sum(((Y-T).^2),3)/R;
            % dummy MMD: assume MMD can lie between 0.5 and 1
            % Take mean over mini-batch.
            N = size(Y,4);
            % divide by N to normalize over mini-batch
            loss = (1/N).*sum(wmse .* sum(meanSquaredError)/N + wnmmd .* mmd);
        end
        
        function [dLdY, varargout] = backwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted MMD-NMSE
            % loss for the training predictions and targets

            wmse = layer.LossWeights(1);
            wnmmd = layer.LossWeights(2);
            mmd = layer.MMD_Metric;
            % Calculate MAE.
            R = size(Y,3);
            meanSquaredError = sum(((Y-T).^2),3)/R;
            % dummy MMD: assume MMD can lie between 0.5 and 1
            % Take mean over mini-batch.
            %R = size(Y,3);
            N = size(Y,4);
            gradMSE = 1*sum((Y-T),3)/R;
            % divide by N to normalize over mini-batch
            dLdY = wmse.*gradMSE./(N);
            dLdW1 = meanSquaredError./(2*N);
            dLdW2 = mmd;
            varargout{1} = dLdW1;
            varargout{2} = dLdW2;
        end
    end
end