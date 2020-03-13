% script to convert all training results into a results table
% 
% get all the files in the current directory
files = what;
allFiles = files.mat;
logFileIndexes = strfind(string(allFiles),'log');
logFiles = allFiles(logFileIndexes{:});

load resultsTable.mat;
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

