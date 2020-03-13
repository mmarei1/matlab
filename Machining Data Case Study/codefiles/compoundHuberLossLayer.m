classdef compoundHuberLossLayer < nnet.layer.RegressionLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        LossWeights
        Delta
        MMD_Metric
    end

    methods
        function layer = compoundHuberLossLayer(name, MMD_Metric, delta)
            % layer = compoundLossLayer(name, MMD_Metric, delta) creates a
            % weighted MK-MMD Huber loss layer. MMD_Metric is a Maximum Mean
            % Discrepancy value equating to the difference between the
            % source and target domain distributions of the pre-trained CNN
            % data. 
            % The Huber loss is defined as 
            %            0.5*(y-f(x))^2 for |y-f(x)| <= delta 
            % and 
            % delta*|y-f(x)| - 0.5*delta^2 otherwise
            % 
            % layer =compoundLossLayer(MMD_Metric, name)
            % additionally specifies the layer name. 

            % Set loss weights
            r = rand(1);
            layer.LossWeights = sort([r,1-r],'descend');
            layer.Delta = delta;
            layer.MMD_Metric = MMD_Metric;
            % Set layer name
            if nargin >= 2
                layer.Name = name;
            end

            % Set layer description
            layer.Description = 'Regression output weighted summation of the MMD and Huber loss';
            
            layer.ResponseNames = {};
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted MMD-NMSE
            % loss for the training predictions and targets

            wLdelta = layer.LossWeights(1);
            wnmmd = layer.LossWeights(2);
            mmd = layer.MMD_Metric;
            delta = layer.Delta;
            
            % Calculate Ldelta, the Huber Loss.
            R = size(Y,3); % number of responses
            N = size(Y,4); % number of observations
            
            huberloss = @(Y,Yhat,wLdelta)sum(wLdelta.*((0.5*(abs(Y-Yhat)<=delta).*(Y-Yhat).^2) + ...
                ((abs(Y-Yhat)>delta).*abs(Y-Yhat)-0.5)))/sum(wLdelta);
            % dummy MMD: assume MMD can lie between 0.5 and 1
            % Take mean over mini-batch.
           
            loss = sum(wLdelta .* sum(huberloss)/N + wnmmd .* mmd);
        end
        
        function [dLdY, dLdWeights dLdDelta] = backwardLoss(layer, Y, T, memory)
            
            N = size(Y,4);
            R = size(Y,3);
            delta = layer.Delta;
            dLdY = zeros(size(Y));      % initialize dLdY
            dhLdY = (1/N).*-(Y-T).*((Y-T)<=delta) + delta.*(sign(Y-T) );
            
            
        end
    end
end