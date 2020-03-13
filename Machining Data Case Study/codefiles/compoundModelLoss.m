function [Loss, gradients, learnables, learnableGradients] = compoundModelLoss(dlnet, dlYPred, X_tgt, X_src, Y_tgt)
%COMPOUNDLOSS calculate the predicted loss for the model using NMSE-NMMD
%   [trainingLoss, modelState] = compoundLoss(dlnet, X_tgt, X_src, Y_tgt) 
%   computes the compound loss on the predictions, based on two different
%   loss components; a Mean Square Error (MSE) component, and a Maximum
%   Mean Discrepancy (MMD) component, given the intermediate representations
%   of X at specific layers of the deep learning model.
%   The overall MMD-NMSE loss is the weighted summation of the losses with
%   a weighting parameter (Alpha), according to the following equation:
%
%           Loss = w1*MSE + \Sigma(1:L) w2*NMMD_1 + w3*NMMD_2
%   
%   The function takes as inputs the following parameters:
%   
%   dlnet: a dlarray object denoting the deep learning model (i.e.
%   converted from a layer graph)
%   X_tgt,Y_tgt: the target domain inputs X_t and the corresponding outputs
%   Y_t
%   X_src:  the source domain inputs X_s.
%   hlayers: the feature representation layers with respect to which the MMD is calculated.
%   For AlexNet, for example, the features are calculated with respect to
%   'fc7' and 'fc8'
%   This implementation assumes only the features can be transferred and,
%   furthermore, that target domain label data exists.
%   
%   The MSE of the target data Y^_t = f(Y_t|X_t) is normalized by the 
%   maximum and minimum target output values. Similarly for the NMMD, the
%   loss term is normalized by the maximum and minimum feature values for
%   each hidden layer output.
%   Created by Mohamed Marei (2019), based on the MATLAB implementation of 
%   MMD by
%   Heiko Strathmann and Dino Sejdinovic (2012)
%  
%------------------------------------------------------------------------------  
        % If model feature representation layers are not specified, find
        % all the model feature representation layers and select the last n
            
        all_hlayers = findCNNFCLayers(dlnet);
        % if more hidden layers than 2, we compute additional weights
            if length(all_hlayers)>2
            % select the final two fc layers if 
            hlayers = dlnet.Layers(all_hlayers(end-1,end)).Name;
            else
            hlayers = dlnet.Layers(all_hlayers(end)).Name;
            end
        
        % Assign learnable weights
        r = rand(1);
        if (r ~= 1-r)
            learnables = sort([r,1-r],'descend');
        else
            r = r+0.1;
            learnables = sort([r,1-r],'descend');
        end
        
        % Calculate the "prediction" output (or forward function) of the model
        %dlYPred = forward(dlnet,X_tgt);
        %dlYPred = sigmoid(dlYPred);
        
        % get the dimensions of the mini-batch and observations
        R = size(Y_tgt,3);
        N = size(Y_tgt,4);

        wmse = learnables(1);
        wnmmd = learnables(2:end);
        % mmd = layer.MMD_Metric;    
        % Calculate MSE 
        Loss_mse = sum(((dlYPred-Y_tgt).^2),3)/R;
        grad_mse = 1*sum((dlYPred-Y_tgt),3)/R;
        [Loss_mmd,grad_mmd] = mmdLoss(X_src, X_tgt);
        % Sum the empirical loss with the MMD loss
        Loss = (1/N).*sum(wmse .* Loss_mse/2 + wnmmd .* Loss_mmd);
        %grad_mmd = 2.*sqrt(abs(mmd));
            % divide by N to normalize over mini-batch
        
        gradients(1) = wmse.*grad_mse./(N);
        gradients(2) = wnmmd.*grad_mmd;
        
        dLdW1 = Loss_mse;
        dLdW2 = Loss_mmd;
        
        % compute output weights
        learnableGradients(1) = dLdW1;
        learnableGradients(2) = dLdW2;
        
        %gradients = dlgradient(Y_tgt,Loss, learnables);
end

