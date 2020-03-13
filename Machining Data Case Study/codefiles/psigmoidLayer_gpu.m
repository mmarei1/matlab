classdef psigmoidLayer_gpu < nnet.layer.Layer
     properties (Learnable)
        % learnable Parameters
         Alpha
         % end of learnable parameters
     end
     properties
         Channels
     end
     methods
        function layer = psigmoidLayer_gpu(channels,name) 
            % Set layer name
            if nargin == 2
                layer.Channels = channels;
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'GPU-enabled parametrized sigmoidLayer with learnable weight Alpha'; 
            layer.Alpha = rand([1 1 channels]);
        end
        % mutli-channel sigmoid function msig
        function Y = msig(X,a,c)
            Y = 1./(1 + exp(a.*(X-c)));
        end
        
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            a = layer.Alpha;
            c = 1;
            Z = msig(X,a,c);
            %Z = gpuArray(gX);
        end
        
%         function [Z, memory] = forward(layer, X)
%             Z = sigmoid(N);
%             memory = 
%         end
        
        function [dLdX, dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer
%             gX = gpuArray(X);
%             gZ = gpuArray(Z);
%             gddZ = gpuArray(dLdZ);
            a = layer.Alpha;
            c = 1;
            dLdX = msig(X,a,c).*(1-msig(X,a,c)).*dLdZ;
            % derivative of Alpha wrt backward outputs
            dLdAlpha = -0.5.*(X -c).*exp(a.*(X-c))./(1+exp(a.*(X-C).^2))*dLdZ;
            
            %gdLdX = gpuArray(dLdX);
        end
    end
end