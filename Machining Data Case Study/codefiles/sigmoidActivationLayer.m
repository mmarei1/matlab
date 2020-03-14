 classdef sigmoidActivationLayer < nnet.layer.Layer
    %% layer = sigmoidActivationLayer(channels,name)
    %
    % Define a sigmoidal activation layer with two input parameters:
    % channels: the number of channels
    % name: the name of the layer.
    % The backward function of this layer is computed via auto
    % differentiation.
    %
    % Created by Mohamed Marei, 2020
    %%
properties (Learnable)
        % learnable Parameters
         %Alpha
         % end of learnable parameters
     end
     properties
         Channels
     end
     methods
        function layer = sigmoidActivationLayer(channels,name) 
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
            Z = sigmoid(X);
            %Z = gpuArray(gX);
        end       
     end
 end