% Script to load all data files as one dataset comprising a table with days
% indexed by chronological order
% step 1: load correct file names from the 
clear; clc;
dirname = "data";
cd("/home/mareim/MATLAB_DRIVE/NewSandbox/CNC Timeseries Data Case Study")
files = ls(dirname);
% UNIX only: files must be converted to cell string
allFiles = strsplit(files);
%filenames = strtrim(string(files));
% only pick files with the CSV extension
filenames = allFiles(contains(allFiles,"csv"));


    for i = 1:size(filenames,2)
        % output two tables, 1 with pre-normalised data and 1 with
        % normalised data
        [tsdata, ndata] = importfile1(filenames{i});
        tsdata.Index(:) = i;
        trimmedname = erase(filenames(i),".csv");
        tsdata.Datetime(:) = datetime(trimmedname);
        % place at the end of a new table
        % if not the first row index
        len = size(ndata,1);
        if i == 1
            firstIndex = 1;
            lastIndex = len;
        else 
            firstIndex = 1 + lastIndex;
            lastIndex = firstIndex + len-1;
        end
        ts_start = firstIndex;
        ts_end = lastIndex;
        lensum(i) = len;
        tsdata_r = movevars(tsdata, ["Index","Datetime"],"Before","Time");
        ydata_vars = {'Index','Datetime','Time','Z'};
        % get only the wanted sensor data along with indexes
        Xdata(firstIndex:lastIndex,:) = tsdata_r;
        Ydata(firstIndex:lastIndex,:) = tsdata_r(:,ydata_vars);
    end

% Before any further processing, remove missing data from X and Y
Xdata = rmmissing(Xdata);
Ydata = rmmissing(Ydata);
%% normalize all predictor and response data before splitting the datasets
% Xdata(:,7:end) = normalize(Xdata(:,7:end),'range',[0,1]);
% Ydata(:,4) = normalize(Ydata(:,4),'range',[0,1]);
%% partition data into training and testing data;
% take 42 sequences for training and 12 sequences for testing

lastIndex = 54;
train_p = 42;
totalIndexes = 1:lastIndex;
idxTrain = sort(totalIndexes(randsample(totalIndexes,train_p)))';
idxTest = sort(setdiff(totalIndexes,idxTrain))';
sections = 2;
%%
% Extract indexes of training data and testing data
for i = 1:size(idxTrain)
    % return all row indexes matching the index i in idxTrain
    rowsTrain = Xdata.Index(:) == idxTrain(i);
    XData_train{i} = Xdata(rowsTrain,:);
    XData_train_cell{i} = table2cell(Xdata(rowsTrain,7:end));
    YData_train{i} = Ydata(rowsTrain,:);
    YData_train_cell{i} = table2cell(Ydata(rowsTrain,4));
end

for i = 1:size(idxTest)
    rowsTest = Xdata.Index(:) == idxTest(i);
    XData_test{i} = Xdata(rowsTest,:);
    XData_test_cell{i} = table2cell(Xdata(rowsTest,7:end));
    YData_test{i} = Ydata(rowsTest,:);
    YData_test_cell{i} = table2cell(Ydata(rowsTest,4));
end

X_Train = vertcat(XData_train{:});
Y_Train = vertcat(YData_train{:});

X_Test = vertcat(XData_test{:});
Y_Test = vertcat(YData_test{:});
%%
[~,ss_train] = sort(cellfun(@length,XData_train_cell),'descend');
XData_train_cell = XData_train_cell(ss_train);
YData_train_cell = YData_train_cell(ss_train);

[~,ss_test] = sort(cellfun(@length,XData_test_cell),'descend');
XData_test_cell = XData_test_cell(ss_test);
YData_test_cell = YData_test_cell(ss_test);

%% calculate mean and standard deviation of x values
xvals = table2array(Xdata(:,7:58));
xvals_norm = (xvals- mu) ./sig;
%%
save('trainingData_cells.mat','X_Train','Y_Train');
save('testingData_cells.mat','X_Test','Y_Test');

save('modelData.mat','XData_train_cell','XData_test_cell','YData_train_cell','YData_test_cell');
sections = 2 +1;
%% Step 1: Plot the sequences as a bar to compare their lengths

figure(1); clf reset;
bar(lensum)
[seqLengths, sortingIndexes] = sort(lensum,'descend');

% sorted sequences help reduce sequence padding
figure(2); clf reset;
bar(seqLengths)
% to sort the data in the table, we need to reshuffle them
sequenceIndexes_train = unique(X_Train.Index);
sequenceIndexes_test = unique(X_Test.Index);

%%
% iterate through all indexes
reorder_train = cell(54,1);
reorder_test = cell(54,1);
ri_train = [];
ri_test = [];
for i = 1:size(seqLengths,2)
    % if current index matches one in training indexes
    if ismember(sortingIndexes(i),sequenceIndexes_train)
        % assign to the ith cell index of reorder_train the indexes
        % i:seqLengths(i)
        reorder_train{i} = sortingIndexes(i)*ones(seqLengths(i),1);
        foundIndexes_train = find(X_Train.Index == sortingIndexes(i));
        ri_train(end+1:end+numel(foundIndexes_train)) = foundIndexes_train;

    % else if current index matches one in training indexes
    elseif ismember(sortingIndexes(i),sequenceIndexes_test)
        
        % assign to the ith cell index of reorder_test the indexes
        % i:seqLengths(i)
        reorder_test{i} = sortingIndexes(i)*ones(seqLengths(i),1);
        foundIndexes_test = find(X_Test.Index == sortingIndexes(i));
        ri_test(end+1:end+numel(foundIndexes_test)) = foundIndexes_test;
    else
        missedIndexes{i} = i;
    end
end

% vertically concatenate cells into vector; this will automatically remove
% empty cells
nonempty_train = reorder_train(~cellfun('isempty',reorder_train));
nonempty_test = reorder_test(~cellfun('isempty',reorder_test));

% sorting sequence- train and test


reorder_train_ne = vertcat(nonempty_train{:});
reorder_test_ne = vertcat(nonempty_test{:});
%%
X_Train_re = X_Train(X_Train.Index(reorder_train_ne),:);
Y_Train_re = Y_Train(Y_Train.Index(reorder_train_ne),:);
X_Test_re = X_Test(X_Test.Index(reorder_test_ne),:);
Y_Test_re = Y_Test(Y_Test.Index(reorder_test_ne),:);

% save these reshuffled datasets
save('trainingData_preproc.mat','X_Train_re','Y_Train_re');
save('testingData_preproc.mat','X_Test_re','Y_Test_re');

%% plot sorted training and testing data in the same plot

figure(3); clf reset;
bar(sequenceIndexes_train,seqLengths(sequenceIndexes_train),'b');
hold on;
bar(sequenceIndexes_test,seqLengths(sequenceIndexes_test),'r');
legend('Training data','Testing data','FontSize',16)
xlabel('Record Index','FontSize',16)
ylabel('Sequence Length','FontSize',16)
title('Training and Testing Data Split','FontSize',18)

%% Prepare training and testing data for FSRNCA and Model Training using Deep Learning

XTrain_re = X_Train(ri_train,1:end);
YTrain_re = Y_Train(ri_train,:);

XTest_re = X_Test(ri_test,1:end);
YTest_re = Y_Test(ri_test,:);

%XTrain = table2array(XTrain_re)';
%YTrain = table2array(YTrain_re)';

% save the reordered datasets
save('training.mat','XTrain_re','YTrain_re');
save('testing.mat','XTest_re','YTest_re');

% 
% XTest = table2array(XTest_re)';
% YTest = table2array(YTest_re)';
%% Set up training and testing 
numFeatures = 52;
numHiddenUnits = 250;
numResponses = 1;
miniBatchSize = 3*400;

layers = [... 
    sequenceInputLayer(numFeatures),...
    lstmLayer(numHiddenUnits,'OutputMode','sequence'),...
    fullyConnectedLayer(180),...
    dropoutLayer(0.4),...
    fullyConnectedLayer(numResponses),...
    regressionLayer,...
    ];
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.0025, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.125, ...
    'Verbose',0, ...
    "ValidationData",{XData_test_cell',YData_test_cell'},...
    "ValidationFrequency",5,...
    'Plots','training-progress');

net1 = trainNetwork(XData_train_cell',YData_train_cell',layers,options);

% Generate predictions
YPred  = predict(net1,XData_test_cell','MiniBatchSize',1);

