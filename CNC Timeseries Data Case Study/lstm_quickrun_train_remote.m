%% load data as "unprocessed" table and cell array form
% safety code to error check
set(0,'DefaultFigureWindowStyle','Docked');
g = gpuDevice;
%%
load TrainingDataNew.mat
load TestingDataNew.mat
load CNC_timeseries_data.mat

%% initialise training option parameters
numFeatures = 52;
numHiddenUnits = 250;
numResponses = 1;
sequenceLength = 600;
fsIndex = 7;
maxSequenceLength = 2879;
maxEpochs = 200;
useFeatureSelection = false;
allFeatures = 1:numFeatures;
% Network size parameters
% sizeFactor - a scalar multiple that increases the width of the LSTM
% layer. Subsequent layers have progressively smaller widths
widthFactor = 8;
% training Parameters
gradientThreshold = 1;
gradientDecayFactor = 0.9;
initialLearnRate = 0.0002;
learnRateDropFactor = 0.8;
learnRateDropPeriod = 10;
maxEpochs = 20;
miniBatchSize = 4;
validationPeriod = 20;
validationPatience = 5;
%% Horizontally concatenate the entire dataset for feature selection
xtrain = horzcat(XTrain_cell{:})';
ytrain = horzcat(YTrain_cell{:})';
xtest = horzcat(XTest_cell{:})';
ytest = horzcat(YTest_cell{:})';
%%
if useFeatureSelection == true
    nca = fsrnca(xtrain,ytrain,'Verbose',1,'FitMethod','exact','Solver','lbfgs')
    % select individual featuree based on an arbitrary parameter value threshold
    vt = 0.1;
    allFeatures = 1:numFeatures;
    selectedFeatures = allFeatures(nca.FeatureWeights > vt)
    figure(1); clf reset;
    plot(nca.FeatureWeights,'ro');
    hold on;
    plot(selectedFeatures,nca.FeatureWeights(selectedFeatures),'bo')
    xlabel('Feature Index')
    ylabel('Feature Weight')
    grid on
    title(['FSRNCA: Selected Features with Weights > ',num2str(vt)],'FontSize',18);
    [featureValues, featureIndexes] = sort(nca.FeatureWeights(nca.FeatureWeights > 0.1),'descend');


    lossvalue = loss(nca,xtest,ytest);

    % steps to improve the performance of the feature selector for sequence data
    % 1. refit the model

    numObs = size(xtrain,1);
    nca2 = refit(nca,'FitMethod','exact','Verbose',1,'Lambda',0.5/numObs)
    lossvalue2 = loss(nca2,xtest,ytest)
    % 2. 
    % Comment out this section to remove feature selection implementation on the dataset
    XTrain_cell_fs = cell(numel(XTrain_cell),1);
    XTest_cell_fs = cell(numel(XTest_cell),1);
    for i = 1:numel(XTrain_cell)
        XTrain_cell_fs{i} = XTrain_cell{i}(selectedFeatures,:);
    end

    for i = 1:numel(XTest_cell)
        XTest_cell_fs{i} = XTest_cell{i}(selectedFeatures,:);
    end
else
    disp('No feature selection approach used. Proceeding with training/testing with full dataset.');
end
%% if feature selection is to be used
%  select appropriate features from training and testing data
%  otherwise, use the dataset as is
if useFeatureSelection == true
    numFeatures = numel(selectedFeatures);
    trainingData = XTrain_cell_fs;
    testingData = XTest_cell_fs;
    fsflag = "-fs";
else
    numFeatures = numel(allFeatures);
    trainingData = XTrain_cell;
    testingData = XTest_cell;
    fsflag = "-nofs";
end
%% Parametrise models and test their performance with different initializers

[layersGlorot, dGlorot] = createLSTMModel("Glorot",numFeatures,widthFactor);
[layersHe, dHe] = createLSTMModel("He",numFeatures,widthFactor);
[layersNN, dNN] = createLSTMModel("narrow-normal",numFeatures,widthFactor);


%lnet = layerGraph(layers);
% 
% options1 = trainingOptions('sgdm', ...
%     'MaxEpochs',maxEpochs, ...
%     'SequenceLength','Shortest', ...
%     'InitialLearnRate',0.000125, ...
%     'LearnRateSchedule','piecewise', ...
%     'Verbose',1, ...
%     "ValidationData",[{XTest_cell};{YTest_cell}],...
%     "ValidationFrequency",validationPeriod,...
%     'Plots','training-progress');

t_options = trainingOptions('adam', ...
'Shuffle','never',...
'ExecutionEnvironment','gpu',...    
'MaxEpochs',maxEpochs, ...
    'SequenceLength','Shortest', ...
    'MiniBatchSize',miniBatchSize,...
    'GradientDecayFactor',gradientDecayFactor,...
    'GradientThreshold',gradientThreshold, ...
    'InitialLearnRate',initialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',learnRateDropPeriod, ...
    'LearnRateDropFactor',learnRateDropFactor, ...
    'Verbose',1, ...
    "ValidationData",[{testingData};{YTest_cell}],...
    "ValidationFrequency",validationPeriod,...
    'ValidationPatience',validationPatience,...
    'Plots','training-progress')

t_options_sgdm = trainingOptions('sgdm', ...
'ExecutionEnvironment','gpu',...    
'MaxEpochs',maxEpochs, ...
    'SequenceLength','Shortest', ...
    'MiniBatchSize',miniBatchSize,...
    'GradientThreshold',gradientThreshold, ...
    'InitialLearnRate',initialLearnRate*10, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',learnRateDropPeriod, ...
    'LearnRateDropFactor',learnRateDropFactor, ...
    'Verbose',1, ...
    "ValidationData",[{testingData};{YTest_cell}],...
    "ValidationFrequency",validationPeriod,...
    'Plots','training-progress');


%% train the models
numFeatures
%

try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
reset(g);
tic;
[netGlorot, infoGlorot] = trainNetwork(trainingData,YTrain_cell,layersGlorot,t_options);
tt_Glorot = toc;
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
reset(g);
tic;
[netHe, infoHe] = trainNetwork(trainingData,YTrain_cell,layersHe,t_options);
tt_He = toc;

try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
reset(g);

tic;
[netNN, infoNN] = trainNetwork(trainingData,YTrain_cell,layersNN,t_options);
tt_NN = toc;

%%
% predict model outputs
YPredGlorot = predict(netGlorot,testingData,'MiniBatchSize',miniBatchSize);
YPredHe = predict(netHe,testingData,'MiniBatchSize',miniBatchSize);
YPredNN = predict(netNN,testingData,'MiniBatchSize',miniBatchSize);
%% Calculate the accuracy threshold of the model based on 5% of ytrain "mode"
accThreshold = 0.05*mode(ytrain);

for i = 1:size(YPredGlorot,1)
    predErrorsGlorot{i} = abs((YPredGlorot{i}-YTest_cell{i}));
    predErrorsHe{i} = abs((YPredHe{i}-YTest_cell{i}));
    predErrorsNN{i} = abs((YPredNN{i}-YTest_cell{i}));
    avgPredErrorGlorot(i) = mean(predErrorsGlorot{i});
    avgPredErrorHe(i) = mean(predErrorsHe{i});
    avgPredErrorNN(i) = mean(predErrorsNN{i});
    % count the instances with an error smaller than the error threshold
    numCorrectGlorot{i} = (abs(predErrorsGlorot{i})) < accThreshold;
    RMSE_vecGlorot(i) = sqrt((mean(YPredGlorot{i}-YTest_cell{i})).^2);
    histsGlorot{i} = histogram(predErrorsGlorot{i});
    numCorrectHe{i} = (abs(predErrorsHe{i})) < accThreshold;
    RMSE_vecHe(i) = sqrt((mean(YPredHe{i}-YTest_cell{i})).^2);
    histsHe{i} = histogram(predErrorsHe{i});
    numCorrectNN{i} = (abs(predErrorsNN{i})) < accThreshold;
    RMSE_vecNN(i) = sqrt((mean(YPredNN{i}-YTest_cell{i})).^2);
    histsNN{i} = histogram(predErrorsNN{i});
end

correctPredictionsGlorot = find(horzcat(numCorrectGlorot{:}));
correctPredictionsHe = find(horzcat(numCorrectHe{:}));
correctPredictionsNN = find(horzcat(numCorrectNN{:}));

totalPredictions = horzcat(YPredGlorot{:});

percentageAccuracyGlorot = numel(correctPredictionsGlorot)/numel(totalPredictions)
percentageAccuracyHe = numel(correctPredictionsHe)/numel(totalPredictions)
percentageAccuracyNN = numel(correctPredictionsNN)/numel(totalPredictions)


RMSE = [mean(RMSE_vecGlorot);
    mean(RMSE_vecHe);
    mean(RMSE_vecNN)];

validationRMSE = [infoGlorot.ValidationRMSE;
    infoHe.ValidationRMSE;
    infoNN.ValidationRMSE;];

idx = all(isnan(validationRMSE));
validationRMSE(:,idx) = [];
%% 
si_test = unique(YTest_re.Index);
for i = 1:numel(si_test)
    figure(i+5); clf reset;
    foundIndexes = find(YTest_re.Index == si_test(i));
    datetimes_i{i} = table2array(YTest_re(foundIndexes,2));
    times_i{i} = table2array(YTest_re(foundIndexes,3));
    dt_i{i} = (datetimes_i{i}.Year + datetimes_i{i}.Month + datetimes_i{i}.Day + (times_i{i}));
    % if there are more timesteps than predictions:
    % truncate the predictions at the correct timestep
    if numel(dt_i{i}) > numel(YPredGlorot{i})
        oldTimes = dt_i{i};
        newTimes = oldTimes(1:numel(YPredGlorot{i}));
        dt_i{i} = newTimes;
    end
    % plot a timeseries based on the sequence indexes]
    predictions = [YTest_cell{i};
        YPredGlorot{i};
        YPredHe{i};
        YPredNN{i}];
    plot((1:numel(YTest_cell{i})),predictions,'LineWidth',2);
    ylim([0.375, 0.525]);
%     hold on
%     plot(1:numel(YTest_cell{i}),YPredGlorot{i},'r');
%     plot(1:numel(YTest_cell{i}),YPredHe{i},'y');
%     plot(1:numel(YTest_cell{i}),YPredNN{i},'k');
%     hold off;
    legend(["Targets" "Glorot" "He" "Narrow Normal"],'Location','Best','FontSize',14);
    xlabel('Time Step','FontSize',16);
    ylabel('Z-axis displacement (normalised)','FontSize',16);
    titlestr = strcat("Predicted Outputs Exp no.",num2str(i));
    title(titlestr,'FontSize',16);
end
%%
figure(22);clf reset;
% change the end of epochs to maxEpochs when re-running the code
epochs = 0:10:10*size(validationRMSE,2)-1;
plot(epochs(5:end),validationRMSE(:,5:end),'LineWidth',2);
xlim([epochs(5), epochs(end)]);
legend(["Glorot" "He" "Narrow Normal"],'Location','Best','FontSize',14);
xlabel('Epoch','FontSize',14);
ylabel('Validation RMSE','FontSize',14);
title('RMSE Convergence Trend for LSTMs with Different Input Weight Initialisation Techniques','FontSize',18)
%% save the models
results = struct('ModelsCompared',{layersGlorot;layersHe;layersNN},...
    'TrainingDetails',{t_options},...
    'Predictions',{YPredGlorot;YPredHe;YPredNN},...
    'ValidationRMSE',{validationRMSE},...
    'TrainingTime',{tt_Glorot;tt_He;tt_NN});
filename = "training-Results-"+ string(datetime('now','Format','yyyy-MM-dd')) + '-adam-' + fsflag +".mat";
save(filename,'results');