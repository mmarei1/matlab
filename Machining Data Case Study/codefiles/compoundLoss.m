%% custom loss function implementation for calculating the MMD-MSE Loss for a regression model
function loss = compoundLoss(model, Y, T)
    wmse = layer.LossWeights(1);
    wnmmd = layer.LossWeights(2);
    mmd = layer.MMD_Metric;
    % Calculate MAE.
    R = size(Y,3);
    meanSquaredError = sum(((Y-T).^2),3)/R;
    % dummy MMD: assume MMD can lie between 0.5 and 1
    % Take mean over mini-batch.
    N = size(Y,4);
    loss = (1/N).*sum(wmse .* sum(meanSquaredError)/N + wnmmd .* mmd);
end