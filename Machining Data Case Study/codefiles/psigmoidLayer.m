 classdef psigmoidLayer < nnet.layer.Layer
     properties (Learnable)
        % learnable Parameters
         Alpha
         % end of learnable parameters
     end
     properties
         % fixed properties
         Channels
         % end of properties
     end
     methods
        function layer = psigmoidLayer(channels,name) 
            % Set layer name
            if nargin == 2
                layer.Channels = channels;
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'parametrized sigmoidLayer with learnable weight'; 
            layer.Alpha = randn([1 1 channels]);
        end
        
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            Z = sigmoid(layer.Alpha .*X);
        end
        
%         function [Z, memory] = forward(layer, X)
%             Z = sigmoid(N);
%             memory = 
%         end
        
        function [dLdX, dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            dLdX = sigmoid(layer.Alpha.*X).*(1-sigmoid(layer.Alpha.* X)).*dLdZ;
            dLdAlpha = min(0,X) .* dLdZ;
            dLdAlpha = sum(sum(dLdAlpha,1),1);
            % Sum over all observations in mini-batch.
            dLdAlpha = sum(dLdAlpha,3);
        end
    end
end