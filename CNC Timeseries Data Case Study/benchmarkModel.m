% Function to independently benchmark models given 
function [percAcc,RMSE] = benchmarkModel(predictions, targets, accThreshold)
YPred = predictions;
YT = targets;
for i = 1:size(YT,2)
    predErrors{i} = abs((YPred{i}-YT{i}));
    avgPredError{i} = mean(predErrors{i});
    % count the instances with an error smaller than the error threshold
    numCorrect{i} = (abs(predErrors{i})) < accThreshold;
    RMSE_v(i) = sqrt((mean(YPred{i}-YT{i}).^2));
    %hists{i} = histogram(predErrorsGlorot_n{i});
end
% output a struct to report the model performance
cp = find(vertcat(numCorrect{:}));
tp = vertcat(YPred{:});
percAcc = numel(cp)/numel(tp);
RMSE = mean(RMSE_v);
end