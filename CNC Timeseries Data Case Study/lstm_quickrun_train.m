%% load data as "unprocessed" table and cell array form
load trainingData.mat
load testingData.mat

%% initialise training option parameters
numFeatures = 52;
numHiddenUnits = 250;
numResponses = 1;
sequenceLength = 600;
fsIndex = 7;
maxSequenceLength = 2879;
maxEpochs = 100;
validationPeriod = 5;
%% Implement FSRNCA here
xtrain = horzcat(XTrain_cell{:})';
ytrain = horzcat(YTrain_cell{:})';
xtest = horzcat(XTest_cell{:})';
ytest = horzcat(YTest_cell{:})';
%%
nca = fsrnca(xtrain,ytrain,'Verbose',1,'FitMethod','exact','Solver','lbfgs')
%%
figure
plot(nca.FeatureWeights,'ro');
xlabel('Feature Index')
ylabel('Feature Weight')
grid on

lossvalue = loss(nca,xtest,ytest)

%% steps to improve the performance of the feature selector for sequence data
% 1. 
% 2. 

%% Parametrise models and test their performance with different initializers

layers1 = [... 
    sequenceInputLayer(numFeatures,'Name','InputLayer1'),...
    lstmLayer(numHiddenUnits*3,'OutputMode','sequence','Name','LSTM_Layer'),...
    fullyConnectedLayer(300,'Name','FC_Layer1'),...
    lstmLayer(300,'Name','LSTM_Layer2','OutputMode','sequence'),...
    fullyConnectedLayer(60,'Name','FC_Layer2'),...
    lstmLayer(60,'Name','LSTM_Layer3','OutputMode','sequence'),...
    fullyConnectedLayer(numResponses,'Name','FC_Layer3'),...
    regressionLayer('Name','RegressionLayer'),...
    ];

layersGlorot = [... 
    sequenceInputLayer(numFeatures,'Name','InputLayer1'),...
    bilstmLayer(numHiddenUnits*2,'OutputMode','sequence','InputWeightsInitializer','glorot','Name','LSTM_Layer'),...
    fullyConnectedLayer(numResponses,'Name','FC_Layer3'),...
    regressionLayer('Name','RegressionLayer'),...
    ];
layersHe = [... 
    sequenceInputLayer(numFeatures,'Name','InputLayer1'),...
    bilstmLayer(numHiddenUnits*2,'OutputMode','sequence','InputWeightsInitializer','he','Name','LSTM_Layer'),...
    fullyConnectedLayer(numResponses,'Name','FC_Layer3'),...
    regressionLayer('Name','RegressionLayer'),...
    ];
layersNN = [... 
    sequenceInputLayer(numFeatures,'Name','InputLayer1'),...
    bilstmLayer(numHiddenUnits*2,'OutputMode','sequence','InputWeightsInitializer','narrow-normal','Name','LSTM_Layer'),...
    fullyConnectedLayer(numResponses,'Name','FC_Layer3'),...
    regressionLayer('Name','RegressionLayer'),...
    ];
lnet = layerGraph(layers);

options1 = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'SequenceLength','Shortest', ...
    'InitialLearnRate',0.000125, ...
    'LearnRateSchedule','piecewise', ...
    'Verbose',1, ...
    "ValidationData",[{XTest_cell};{YTest_cell}],...
    "ValidationFrequency",validationPeriod,...
    'Plots','training-progress');

options = trainingOptions('adam', ...
    'MaxEpochs',60, ...
    'SequenceLength','Shortest', ...
    'GradientDecayFactor',0.8,...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.00025, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',15, ...
    'LearnRateDropFactor',0.25, ...
    'Verbose',1, ...
    "ValidationData",[{XTest_cell};{YTest_cell}],...
    "ValidationFrequency",5,...
    'Plots','training-progress');

%% train the model
[netGlorot, infoGlorot] = trainNetwork(XTrain_cell,YTrain_cell,layersGlorot,options);
[netHe, infoHe] = trainNetwork(XTrain_cell,YTrain_cell,layersHe,options);
[netNN, infoNN] = trainNetwork(XTrain_cell,YTrain_cell,layersNN,options);

%%
% predict model outputs
YPredGlorot = predict(netGlorot,XTest_cell,'MiniBatchSize',1);
YPredHe = predict(netHe,XTest_cell,'MiniBatchSize',1);
YPredNN = predict(netNN,XTest_cell,'MiniBatchSize',1);
%%
accThreshold = 0.05;

for i = 1:size(YPredGlorot,1)
    predErrors{i} = abs((YPredGlorot{i}-YTest_cell{i}));
    avgPredError(i) = mean(predErrors{i});
    % count the instances with an error smaller than the error threshold
    numCorrect{i} = (abs(predErrors{i})) < accThreshold;
    RMSE_vec(i) = sqrt((mean(YPredGlorot{i}-YTest_cell{i})).^2);
    hists{i} = histogram(predErrors{i});
end

correctPredictions = find(horzcat(numCorrect{:}));
totalPredictions = horzcat(YPredGlorot{:});

percentageAccuracy = numel(correctPredictions)/numel(totalPredictions)
RMSE = mean(RMSE_vec)

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
    if numel(dt_i{i}) > numel(YPred{i})
        oldTimes = dt_i{i};
        newTimes = oldTimes(1:numel(YPred{i}));
        dt_i{i} = newTimes;
    end
    % plot a timeseries based on the sequence indexes]
    predictions = [YTest_cell{i};
        YPredGlorot{i};
        YPredHe{i};
        YPredNN{i}];
    plot((1:numel(YTest_cell{i})),predictions);
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
epochs = 0:validationPeriod:maxEpochs;
plot(epochs,validationRMSE);
legend(["Glorot" "He" "Narrow Normal"],'Location','Best','FontSize',14);
xlabel('Epoch','FontSize',14);
ylabel('Validation RMSE','FontSize',14);
title('RMSE Convergence','FontSize',18)