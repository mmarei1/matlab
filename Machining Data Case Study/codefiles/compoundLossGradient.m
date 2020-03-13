% TODO; Fix loss computation!!!!
function [dLdY, varargout] = compoundLossGradient(Loss, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted MMD-NMSE
            % loss for the training predictions and targets

            wmse = layer.LossWeights(1);
            wnmmd = layer.LossWeights(2);
            mmd = layer.MMD_Metric;
            % Calculate MAE.
            R = size(Y,3);
            meanSquaredError = sum(((Y-T).^2),3)/R;
            % dummy MMD: assume MMD can lie between 0.5 and 1
            % Take mean over mini-batch.
            %R = size(Y,3);
            N = size(Y,4);
            gradMSE = 1*sum((Y-T),3)/R;
            % divide by N to normalize over mini-batch
            dLdY = wmse.*gradMSE./(N);
            dLdW1 = meanSquaredError./(2*N);
            dLdW2 = mmd;
            varargout{1} = dLdW1;
            varargout{2} = dLdW2;
end