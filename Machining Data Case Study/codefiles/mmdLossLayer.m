classdef mmdLossLayer < nnet.layer.Layer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Learnable)
        Weights
    end
    
    methods
        function layer = mmdLossLayer(name,inputLayers,sourceDomainData)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            layer.Name = name;
            layer.Description = "Maximum Mean Discrepancy Layer calculated on Fully Connected Layers: " + inputLayers.Name;
            layer.InputLayers = inputLayers;
            
        end
        
        function output = predict(layer,inputLayers,X_tgt,X_src)
            %PREDICT compute the MMD
            %   compute the MMD loss of the target and source data using
            %   the layer 
            [output,~] = mmdLoss(X_src,X_tgt,dlnet,inputLayers);
        end
        
        function outputArg = backward(layer,forwardInput)
            outputArg 
        end
    end
end

