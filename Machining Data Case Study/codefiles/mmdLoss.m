% mmdLoss Compute the loss term and gradient of the MMD of the feature 
% representations of dlnet model layers (hlayers)
%
% [Loss, gradient] = mmdLoss(X_src,X_tgt,dlnet,hiddenLayers) computes the Maximum Mean Discrepancy (MMD)
% of the layer-wise feature representations of source domain data, 
% X_src, and target domain data, X_tgt. 
% The function requires as inputs the dlnet and specific layer names (hlayers) whose
% feature representations are input into the MMD computation.
% 
%   Created by Mohamed Marei (2019), based on the MATLAB implementation of 
%   MMD by
%   Heiko Strathmann and Dino Sejdinovic (2012)
function [Loss, gradient] = mmdLoss(X_src,X_tgt,N,dlnet,hlayers)
 % Extract features with respect to specific model layer for input
        % into MMD algorithm
        % create mini-batches of the source and target domain data
            d_sz = size(cell2mat(table2cell(X_src(1,'input'))));
            src_sh = reshape(cell2mat(table2cell(X_src(1:N,1))),[d_sz N]);
            tgt_sh = reshape(cell2mat(table2cell(X_tgt(1:N,1))),[d_sz N]);
            %whos src_sh tgt_sh
            X_tgt = dlarray(single(tgt_sh),'SSCB');
            X_src = dlarray(single(src_sh),'SSCB');
%         end
        % 
        dlnet = dlnetwork(dlnet);
        X_tgt_frs = forward(dlnet,X_tgt,'Outputs',hlayers(2).Name);
        X_src_frs = forward(dlnet,X_src,'Outputs',hlayers(2).Name);
        % type-casting the features so the dlarray structure is disregarded
        X_tgt_frs = cast(X_tgt_frs,'like',0.5);
        X_src_frs = cast(X_src_frs,'like',0.5);
        % Calculate MMD in 2 steps:
        % Step 1: find optimal kernel combination between source and target
        % data representations within sigmas range specified as
        % sigmas = 2.^[-5:1;15];
        [bg] = opt_kernel_comb(X_src_frs,X_tgt_frs);
        % Step 2: calculate mmd from optimal kernel from source, target, optimal
        % kernels and corresponding kernel weights (which we assume to be 1)
        [sigma1,tmean,tvar,tcdf] = mmd_linear_combo(X_src_frs,X_tgt_frs,bg,1);
        % tmean is calculated in terms of MMD^2; loss is the square root of
        % MMD
        Loss = sqrt(tmean);
        % the gradient of the mmd loss w.r.t. learned parameters is found by 
        gradient = 1/2*sqrt(Loss);
end