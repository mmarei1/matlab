classdef logsigRegressionLayer < nnet.layer.RegressionLayer
    %logsigRegressionLayer - Logistic Sigmoid Regression Layer
    %   constructs a Logistic Sigmoid Regression Layer
    
    properties
    %    Property1
    end
    
    methods
        function layer = logsigRegressionLayer(name)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            layer.Name = name;
            %layer.Type = 'RegressionLayer';
            layer.Description = "LogSigmoid Regression Layer";
        end
        
        function loss = forwardLoss(layer,Y,T)
            %FORWARDLOSS Compute the forward loss of the layer 
            %  The loss of this layer is computed by implementing the
            %  logsig function onto the prediction outputs
            
            % "squashing function" to squash range of y outputs
            ce = -T.*log(Y) - (1-T).*log(1-Y);
            
            R = size(Y,1);
            % Compute the loss of the logsig error

            loss = abs(sum(ce))/R;
        end
        
        function dLdY = backwardLoss(layer,Y,T)
            %BACKWARDLOSS Computes the backward loss of the targets w.r.t. the predictions Y 
            %   Detailed explanation goes here
            YPred = logsig(Y-T);
            R = size(Y,1);
            dLdY = dlogsig(YPred,T)/R;
        end
        
    end
end

