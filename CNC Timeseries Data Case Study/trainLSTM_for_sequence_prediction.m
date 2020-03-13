%load CNC_timeseries_data.mat
load TrainingDataHPC.mat
load TestingDataHPC.mat


%%
numFeatures = 52;
numHiddenUnits = 250;
numResponses = 1;
sequenceLength = 600;
fsIndex = 7;
maxSequenceLength = 2879;
%%
% take only the required column data from the stored datasets
% XTrain = table2array(XTrain_re(:,fsIndex:fsIndex-1+numFeatures))';
% YTrain = table2array(YTrain_re(:,4))';
% 
% XTest = table2array(XTest_re(:,fsIndex:fsIndex-1+numFeatures))';
% YTest = table2array(YTest_re(:,4))';

XTrain = table2array(XTrain_re(:,fsIndex:fsIndex-1+numFeatures))';
YTrain = table2array(YTrain_re(:,4))';

XTest = table2array(XTest_re(:,fsIndex:fsIndex-1+numFeatures))';
YTest = table2array(YTest_re(:,4))';

% arrange sequences into cells for training
% sort on XTrain_cell
[sv_train,I_train] = sort(cellfun(@length,XTrain_cell),'descend');
sequenceIndexes_Cl_train = I_train;

% sort on XTest_cell
[sv_test,I_test] = sort(cellfun(@length,XTest_cell),'descend');
sequenceIndexes_Cl_test = I_test;

sequenceIndexes = vertcat(sequenceIndexes_Cl_train,sequenceIndexes_Cl_test);

for i = 1:54
    if ismember(sequenceIndexes(i),unique(XTrain_re.Index))
       foundIndexes = find(XTrain_re.Index == sequenceIndexes(i));
       XTrain_cell{i} = XTrain_re{foundIndexes,fsIndex:fsIndex-1+numFeatures}';
       YTrain_cell{i} = YTrain_re{foundIndexes,4}';
       lensum_train(i,:) = [i,size(XTrain_re(foundIndexes,:),1)];
    elseif ismember(sequenceIndexes(i),unique(XTest_re.Index))
       foundIndexes = find(XTest_re.Index == sequenceIndexes(i));
       XTest_cell{i} = XTest_re{foundIndexes,fsIndex:fsIndex-1+numFeatures}';
       YTest_cell{i} = YTest_re{foundIndexes,4}';
       lensum_test(i,:) = [i,size(XTest_re(foundIndexes,:),1)];
    end
end

% [sequenceLengths_train, si_train] = sort(lensum_train(:,2),'descend');
% [sequenceLengths_test, si_test] = sort(lensum_test(:,2),'descend');
% 
% si_train = si_train(sequenceLengths_train>0);
% si_test = si_test(sequenceLengths_test>0);

% delete empty cells
XTrain_cell = XTrain_cell(~cellfun('isempty',XTrain_cell));
YTrain_cell = YTrain_cell(~cellfun('isempty',YTrain_cell));

XTest_cell = XTest_cell(~cellfun('isempty',XTest_cell));
YTest_cell = YTest_cell(~cellfun('isempty',YTest_cell));

%% Anomalous data extraction
anom_XTest = cellfun(@isnan,XTest_cell,'UniformOutput',false);

% Re-compile XTest without missing indexes
for i = 1:size(XTest_cell,2)
    indexesToKeep = ~anom_XTest{i};
    indVector = any(indexesToKeep); 
    %keptIndexes = repmat(indexesToKeep,[52,1]);
    arrayToExtract = XTest_cell{i};
    vecToExtract = YTest_cell{i};
    XTest_cell_grid{i} = arrayToExtract(indexesToKeep);
    YTest_cell_vec{i} = vecToExtract(indVector);
    %lengthOfvec = numel(XTest_cell{i})/52;
    XTest_cell_grid{i} = reshape(XTest_cell_grid{i},52,[]);
end

XTest_cell = XTest_cell_grid;
YTest_cell = YTest_cell_vec;

[~,order] = sort(cellfun(@length,XTest_cell),'descend');
XTest_cell = XTest_cell(order);
YTest_cell =YTest_cell(order);
%%
miniBatchSize = size(XTrain,2);
% attempt to 
layers = [... 
    sequenceInputLayer(numFeatures,'Name','InputLayer1'),...
    lstmLayer(numHiddenUnits*2,'OutputMode','sequence','Name','LSTM_Layer'),...
    fullyConnectedLayer(240,'Name','FC_Layer1'),...
    lstmLayer(240,'Name','LSTM_Layer2','OutputMode','sequence'),...
    fullyConnectedLayer(60,'Name','FC_Layer2'),...
    lstmLayer(60,'Name','LSTM_Layer3','OutputMode','sequence'),...
    fullyConnectedLayer(numResponses,'Name','FC_Layer3'),...
    regressionLayer('Name','RegressionLayer'),...
    ];
lnet = layerGraph(layers);

options = trainingOptions('adam', ...
    'MaxEpochs',180, ...
    'SequenceLength','Shortest', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.0005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',1, ...
    "ValidationData",[{XTest_cell};{YTest_cell}],...
    "ValidationFrequency",5,...
    'Plots','training-progress');
%% Calculate the feature dimension of each sequence
% Find anomalies (NaN values in testing data before training
%anomalies_XTest = cellfun(isNaN,XTest_cell,'UniformOutput',false)

[net1, info1] = trainNetwork(XTrain_cell,YTrain_cell,layers,options);

%% Predict outputs 
YPred = predict(net1,XTest_cell,'MiniBatchSize',100);
%%
accThreshold = 0.05;

for i = 1:size(YPred,1)
    predErrors{i} = abs((YPred{i}-YTest_cell{i}));
    avgPredError(i) = mean(predErrors{i});
    % count the instances with an error smaller than the error threshold
    numCorrect{i} = (abs(predErrors{i})) < accThreshold;
    RMSE_vec(i) = sqrt((mean(YPred{i}-YTest_cell{i})).^2);
    hists{i} = histogram(predErrors{i});
end

correctPredictions = find(horzcat(numCorrect{:}));
totalPredictions = horzcat(YPred{:});

percentageAccuracy = numel(correctPredictions)/numel(totalPredictions)
RMSE = mean(RMSE_vec)

%%
for i = 1:numel(si_test)
    figure(i+5); clf reset;
    foundIndexes = find(YTest_re.Index == si_test(i));
    datetimes_i{i} = table2array(YTest_re(foundIndexes,2));
    times_i{i} = table2array(YTest_re(foundIndexes,3));
    dt_i{i} = (datetimes_i{i}.Year + datetimes_i{i}.Month + datetimes_i{i}.Day + (times_i{i}));
    % if there are more timesteps than predictions:
    % truncate the predictions at the correct timestep
    if numel(dt_i{i}) > numel(YPred{i})
        oldTimes = dt_i{i};
        newTimes = oldTimes(1:numel(YPred{i}));
        dt_i{i} = newTimes;
    end
    plot(dt_i{i},YTest_cell{i},'b');
    hold on
    plot(dt_i{i},YPred{i},'r');
    hold off;
    legend('Targets','Predictions','FontSize',14);
    xlabel('Time (HH:MM:SS)','FontSize',16);
    ylabel('Z-axis displacement (normalised)','FontSize',16);
    titlestr = strcat("Predicted Outputs Exp no.",num2str(i));
    title(titlestr,'FontSize',16);
end
%% Compare this methodology to other proposed approaches from the literature
%% Why is the configuration of this network suitable for the problem being investigateed?
%% Can the problem parameters be related to the network design?
%% Can this be extended to other problem?
%% 
