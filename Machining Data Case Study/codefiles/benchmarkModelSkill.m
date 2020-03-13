function [MAE,rmse,accuracy_10,accuracy_20,accuracy_30] = benchmarkModelSkill(YPred,YActual)
%benchmarkModelSkill function computes the MAE, RMSE and accuracy details of a
% model based on the predictions and targets provided as inputs.
%   Inputs: YPred (predictions); YActual (targets)
%   Outputs: MAE, RMSE and accuracy within given thresholds, 10%, 20% and
%   30% of a given value.
%   Created by Mohamed Marei, 2019
%_______________________________________________________________________________
    predictionError = abs(YActual - YPred);
    MAE = mean(predictionError)
    squares = predictionError.^2;
    rmse = sqrt(mean(squares))
    
    thr_10 = 0.10;
    thr_20 = 0.20;
    thr_30 = 0.30;
    numCorrect_10 = sum(predictionError < thr_10);
    numCorrect_20 = sum(predictionError < thr_20);
    numCorrect_30 = sum(predictionError < thr_30);
    
    numValidationImages = numel(YActual);
    
    accuracy_10 = numCorrect_10/numValidationImages
    accuracy_20 = numCorrect_20/numValidationImages
    accuracy_30 = numCorrect_30/numValidationImages

end

