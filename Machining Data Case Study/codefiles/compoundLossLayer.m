classdef compoundLossLayer < nnet.layer.RegressionLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        LossWeights
        % Mini-batch source and target domain examples
        X_src
        X_tgt
    end

    methods
%         function mmdLoss = computeMMDLoss(layer,larray, ds_source, idx)
%             % calculate the MMD loss value based on the source domain
%             % feature vector representation
%             %
%             % Step 1: convert mini-batch of source domain data into correct
%             % format
%             ds_source = dlarray();
%             ds_target = dlarray();
%             
%         end
        
        function layer = compoundLossLayer(name, larray, ds_source)
            % layer = compoundLossLayer(MMD_Metric, name) creates a
            % weighted MK-MMD MSE loss layer. MMD_Metric is a Maximum Mean
            % Discrepancy value equating to the difference between the
            % source and target domain distributions of the pre-trained CNN
            % data.
            % 
            % layer = compoundLossLayer(MMD_Metric, name)
            % additionally specifies the layer name. 

            % Set loss weights
            r = rand(1);
            layer.LossWeights = sort([r,1-r],'descend');
            %layer.MMD_Metric = MMD_Metric;
            % Set layer name
            if nargin == 2
                layer.Name = name;
            end

            % Set layer description
            layer.Description = 'Regression output weighted summation of the MMD and MSE loss';
            layer.ResponseNames = {};
            
        end
        
        function [Loss, gradient] = computeMMDLoss(dlnet,hlayers,X_src,X_tgt)
        % Extract features with respect to specific model layer for input
        % into MMD algorithm
        
        X_tgt = dlarray(single(X_tgt),'SSCB');
        X_src = dlarray(single(X_src),'SSCB');
        
        X_tgt_frs = forward(dlnet,X_tgt,'Outputs',hlayers);
        X_src_frs = forward(dlnet,X_src,'Outputs',hlayers);
        % Calculate MMD in 2 steps:
        % Step 1: find optimal kernel combination between source and target
        % data representations within sigmas range specified as
        % sigmas = 2.^[-5:1;15];
        [bg] = opt_kernel_comb(X_src_frs,X_tgt_frs);
        % Step 2: calculate mmd from optimal kernel from source, target, optimal
        % kernels and corresponding kernel weights (which we assume to be 1)
        [~,tmean,~,~] = mmd_linear_combo(X_src_frs,X_tgt_frs,bg,1);
        % tmean is calculated in terms of MMD^2; loss is the square root of
        % MMD
        Loss = sqrt(tmean);
        % the gradient of the mmd loss w.r.t. learned parameters is found by 
        gradient = 1/2*sqrt(Loss);
        end
        
        function [loss,terms] = forwardLoss(layer, Y, T, hlayers, X_src, X_tgt)
            % loss = forwardLoss(layer, Y, T) returns the weighted MMD-NMSE
            % loss for the training predictions and targets

            wmse = layer.LossWeights(1);
            wnmmd = layer.LossWeights(2);
            % mini-batch of data retrieved from the layer
            R = size(Y,3);
            N = size(Y,4);
            % compute MMD
            [mmd,gradMMD] = computeMMDLoss(layer, hlayers, X_src, X_tgt);
            %layer.MMD_Metric;
            % Calculate 1/2 MSE.
            meanSquaredError = 0.5.*sum(((Y-T).^2),3)/R;
            % Calculate overall loss by summming MSE + MMD: assume MMD can lie between 0.5 and 1
            % Take mean over mini-batch.
            loss = (1/N).*sum(wmse .* meanSquaredError + wnmmd .* mmd);
            terms(1) = meanSquaredError;
            terms(2) = mmd;
            terms(3) = gradMMD;
        end
        
        function [gradMSE, gradMMD, gradientsLearnables] = backwardLoss(layer, hlayers, Y, T ,terms)
            % compute the MMD gradients
            gradMSE = 1*sum((Y-T),3)/R;
            gradientsLearnables(1) = terms(1);
            gradientsLearnables(2) = terms(2);            
            gradMMD = terms(3);
        end
    end
end