% function to compute model loss and gradient for custom training loop
% Loss and gradient are computed from the weighted summation of MSE and MMD
% as:
% Loss = w1*mse + w2*mmd;
% grad = w1*grad_mse + w2*grad_mmd;
%
% The MMD loss and its gradient are computed using the mmdLossD2 function,
% which takes as input the source and target feature representations, 
% sdf and tdf, respectively.
% The MMD computation is adapted from MATLAB code by Heiko Strathmann and Dino
% Sejidnovic, 2012.
%
% Created by Mohamed Marei, 2020
function [loss, gradient] = modelGradients(Y,T,sdf,tdf,weights)
    % compute the predictions w.r.t. the dlnetwork model
    wmse = weights(1);
    wmmd = weights(2);
    R = size(Y,3);          % number of responses
    N = size(Y,4);          % number of observations
    dlYPred = Y;
    % calculate the RMSE loss component and its gradient
    loss_rmse = (1/N).*sqrt(mse(Y,T));
    grad_mse = 2*wmse*N*(Y-T);
    % calculate the transfer loss and its gradient
    [loss_mmd,grad_mmd] = mmdLossD2(sdf,tdf);
    % sum the loss and gradient
    loss = wmse*loss_rmse + wmmd*loss_mmd;
    gradient = wmse*grad_mse + wmmd*grad_mmd;
end