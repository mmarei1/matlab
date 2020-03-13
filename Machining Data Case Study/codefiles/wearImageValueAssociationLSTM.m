% Script to associate the wear images to corresponding wear measurements

% For images labelled 0-0.2-0.6-2-6M we need to give some initial
% estimates close to zero, as we don't expect these values to significantly
% impact the prediction accuracy of the model in later (more critical)
% stages

totalDataPoints = [zeros(25,5) table2array(machiningDataTable(:,2:end))];

for i = 1:25
    count = count + 5 + size(totalDataPoints(i,:)>0) ,2)
end


earlyCuttingDistances = repmat([0 0.2 0.6 2 6],[25,1]);

