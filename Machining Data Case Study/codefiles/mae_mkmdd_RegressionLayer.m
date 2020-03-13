classdef mae_mkmdd_RegressionLayer < nnet.layer.Layer
    % Example custom regression layer with mean-absolute-error loss.
    properties (Learnable)
        Lambda
    end
    
    methods
        function layer = mae_mkmdd_RegressionLayer(numChannels, name, loss_toggle)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer with MK-MMD penalty and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'Mean Absolute Error with MK-MMD loss';
            if loss_toggle == 0
                layer.MKMMD_loss_toggle = 'disabled';
                layer.Lambda = zeros([1 1 numChannels])
            else
                layer.MKMMD_loss_toggle = 'enabled';
                layer.Lambda = randn([1 1 numChannels])
            end
            
            
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            % Calculate MAE.
            R = size(Y,3);
            meanAbsoluteError = sum(abs(Y-T),3)/R;
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sum(meanAbsoluteError)/N + Layer.Lambda.*mmd();
        end
    end
end