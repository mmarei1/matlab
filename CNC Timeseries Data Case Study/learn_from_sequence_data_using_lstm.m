% load data files for use with several Deep Learning apps
clear; clc; close all;

% change the file path to load a different csv file
datapath = "C:\Users\Mareim\OneDrive - Coventry University\Data\Data on CNC temp\data\2017-03-06.csv";

dateOfDay = extractBetween(datapath,"data\",".csv");

[tsdata, ndata] = importfile1(datapath, [2, Inf]);
interval = 1:size(tsdata,1);
tsdata_interval = tsdata(interval,:);
tt1 =  table2timetable(tsdata_interval);
% fix date and time
t = tsdata.Time;
d = datetime(dateOfDay);
t.Year = d.Year;
t.Month = d.Month;
t.Day = d.Day;

% Response and predictor (temperature sensor feeds) variables (pre-normalised)


X_pre = tsdata.X(interval);
Y_pre = tsdata.Y(interval);
Z_pre = tsdata.Z(interval);


predictors_pre = tsdata{interval,5:end};% 
ts_1 = tsdata(interval,1);
% Response and predictor variables (post-normalised)

Xval = ndata.X;
Yval = ndata.Y;
Zval = ndata.Z;

%predictors = [interval, ndata{:,4:end}];
predictors = ndata{:,4:end};
% take a sample sensor as input (column 1:48 specifying sensor location in
% 6x8 grid; column 49:52 specifying sensor 1:4 around the spindle)
pred1 = predictors(:,6);

% Plot figures
ts_interval = tsdata.Time(interval);
d = datetime(dateOfDay);
ts_interval.Year = d.Year;
ts_interval.Month = d.Month;
ts_interval.Day = d.Day;

%plotdata();
%% Split the dataset into training and testing data
% Create three two-hour-long sampling intervals, with random starting points
numSamplingIntervals = 3;
samplingMinutes = 240;
startTimes = sort(randsample(24,numSamplingIntervals)*120);
% assign random start points for sampling 
startTimes = startTimes - 4*randsample(60,3)
for i = 1:size(startTimes,1)
    if startTimes(i) < 0
        startTimes = startTimes*-1
    end
end
endTimes = startTimes(:) + samplingMinutes
%% Generate a subset of testing samples
% add an index to predictors - 
predictors(:,1);

for i = 1:numSamplingIntervals
    % calculate sampling interval duration
    intervalPoints = startTimes(i):endTimes(i); 
    ezs = [intervalPoints', Zval(intervalPoints')];
    newStartPoint = startTimes(i);
    % starting at the first point in the interval + the interval
    newEndPoint = newStartPoint + intervalPoints(end);
    eps = [predictors(startTimes(i):endTimes(i),:)];
    % set new start index
    startIndex = 1+(i-1)*samplingMinutes
    extZSamples(startIndex:startIndex+samplingMinutes,:) = ezs;
    extPredSamples(startIndex:startIndex+samplingMinutes,:) = eps;
    %extPredSample() = eps;
    %Ztest((i-1)*240+1:i*240+i,1) = extSample;
end

%%
predictors_test =  extPredSamples;
predictors_train = predictors;
indexesToRemove = extZSamples(:,1);
% remove indexes in indexesToRemove
predictors_train(indexesToRemove,:) =  [];
% similarly, remove indexes in Zvals_train
Zvals_test = extZSamples;
Zvals_train = [interval', Zval];
Zvals_train(indexesToRemove,:) = [];

figure(3); clf reset;
testIndeces = Zvals_test(:,1);
trainIndeces = Zvals_train(:,1);
plot(t(testIndeces),Zvals_test(:,2),'b');
hold on;
plot(t(trainIndeces),Zvals_train(:,2),'r');
legend('Testing data','Training data');

%% Approach 1: Feature selection for regression using Neighborhood Component
% Analysis
% Create a fsrnca model to weight each feature based on a predefined
% threshold Lambda, equal to 0.5/numObservations
lambda = 0.5/size(interval,2);
mdl = fsrnca(predictors_pre, Zval,'Verbose',1,'FitMethod','Exact','Solver','lbfgs');
[fi,fj,fs]  = find(mdl.FeatureWeights > 0.02);
%% plot these results
figure(2); clf reset;
plot(mdl.FeatureWeights,'ro');
hold on;
scatter(fi,mdl.FeatureWeights(fi),'MarkerFaceColor','b');
legend('Feature','Weight > 0.02');
xlabel('Feature Index');
ylabel('Feature Weight');
grid on;
title('Feature Selection for Regression using Neighborhood Component Analysis: Feature Weights');

%% Approach 2: Use the Diagnostic Feature Designer App to generate condition indicators
%% Approach 3: Use a simple LSTM layer to predict the output of the z-axis displacement taking in the file ensemble datastore as input
numFeatures = 52;
numHiddenUnits = 50;
numResponses = 1;
layers = [... 
    sequenceInputLayer(numFeatures),...
    lstmLayer(numHiddenUnits,'OutputMode','sequence'),...
    fullyConnectedLayer(50),...
    dropoutLayer(0.5),...
    fullyConnectedLayer(numResponses),...
    regressionLayer;
    ];
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%%
% Designate the training and testing data and pre-train the model
XTrain = predictors_train';
YTrain = Zvals_train(:,2:end)';

XTest = predictors_test';
YTest = Zvals_test(:,2:end)';

net = trainNetwork(XTrain, YTrain, layers, options);

YPred = predict(net,XTest,'MiniBatchSize',1);
%% Plot the results
testData = [{t(testIndeces)},{YTest}];
predData = [{t(testIndeces)},{YPred}];

% Create a figure to show the prediction outputs
figure(4); clf reset;
plot(t,Zval(:,1));
hold on
scatter(testData{:,1},testData{:,2},'r');
scatter(predData{:,1},predData{:,2},'g');
hold off
legend('Training Data','Testing Data','Predicted Outputs','Location','Best','FontSize',16)
ylabel('Displacement in z-axis (mm)','FontSize',16);
xlabel('Time (hh:mm:ss)','FontSize',16);
title('LSTM z-axis CNC Machine Centre Displacement Prediction on 24-hour Sequence','FontSize',18);

