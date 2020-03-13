 classdef mysigmoidLayer < nnet.layer.Layer
     properties (Learnable)
        % learnable Parameters
         %Alpha
         % end of learnable parameters
     end
     properties
         Channels
     end
     methods
        function layer = mysigmoidLayer(channels,name) 
            % Set layer name
            if nargin == 2
                layer.Channels = channels;
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'mysigmoidLayer'; 
            %layer.Alpha = 
        end
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            Z = sigmoid(X);
        end
        
%         function [Z, memory] = forward(layer, X)
%             Z = sigmoid(N);
%             memory = 
%         end
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            dLdX = sigmoid(X).*(1-sigmoid(X)).*dLdZ;
            
        end
    end
end