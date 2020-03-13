 classdef sigmoidLayer_gpu < nnet.layer.Layer
     properties (Learnable)
        % learnable Parameters
         %Alpha
         % end of learnable parameters
     end
     properties
         Channels
     end
     methods
        function layer = sigmoidLayer_gpu(channels,name) 
            % Set layer name
            if nargin == 2
                layer.Channels = channels;
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'GPU-enabled sigmoidLayer'; 
           % layer.Alpha = rand([1 1 channels]);
        end
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            Z = sigmoid(X,1);
            %Z = gpuArray(gX);
        end
        
%         function [Z, memory] = forward(layer, X)
%             Z = sigmoid(N);
%             memory = 
%         end
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer
%             gX = gpuArray(X);
%             gZ = gpuArray(Z);
%             gddZ = gpuArray(dLdZ);
            dLdX = Z.*(1-Z).*dLdZ;
            %gdLdX = gpuArray(dLdX);
        end
    end
end