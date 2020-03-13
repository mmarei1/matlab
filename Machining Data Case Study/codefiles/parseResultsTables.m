% script to convert all training results into a results table
load pretrained-log-alexnet-29-Nov-2018.mat;

it = struct2table(results,'AsArray',1);
it = addvars(it,1,'NewVariableNames','id','Before','NetworkName')
% the new table row should only contain a subset of the variables
% -network name
% training time
% maximum accuracy
% training details

performanceTable = struct2table(it.Performance);

it = removevars(it,{'Performance','validationData','Predictions','TrainingOptions'});

it = table(it,performanceTable);

resultsTable = splitvars(it,{'it','performanceTable'});

% insert an index variable
save resultsTable.mat

writetable(resultsTable,'classification-results.txt','Delimiter',' ','WriteVariableNames',true);

