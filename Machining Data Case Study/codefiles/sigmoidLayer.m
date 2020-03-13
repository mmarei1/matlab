 classdef sigmoidLayer < nnet.layer.Layer
     properties (Learnable)
         Alpha
     end
     
     methods
        function layer = sigmoidLayer(name) 
            % Set layer name
            if nargin == 1
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'sigmoidLayer'; 
            layer.Alpha = 1;
        end
        
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            Z = exp(X)./(exp(X)+1);
        end
        
        function [Z,memory] = forward(layer,X)
            Z = exp(X)./(exp(X)+1);
            memory = sqrt((predict(layer,X) - Z).^2);
        end
        
        function [dLdX, dLdW] = backward(layer,X,Z, memory,dLdZ)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            dLdX = X.*(1-X) .* dLdZ;
            dLdW = memory*layer.Alpha;
            
        end
    end
end