%% Machine Cutting Tool Wear RUL Prediction using Component Degradation Models
% Created by Mohamed Marei, 14/08/2018


%% Data and Experiment Description
% The dataset contains 25 run-to-failure experiments of a milling tool,
% with the measured values of flank tool wear length recorded every 20 metres,
% between values of 0 m and 240 m.
% The milling tool is considered "fully worn" at a wear value >= 0.4 mm.
% The data also contains a set of cutting parameters for each experiment,
% such as cutting depth and width, cutting speed, feedrate per teeth,
% feedrate, and radial run-out of cutting tool.
% The objective is to use the available data to build a predictive model of
% tool wear given the cutting lenght. Also, we are interested in
% determining which of the provided data has the most significant impact on
% the tool life, i.e. which of these conditions causes the tool to degrade
% the fastest.

% To do this, we will attempt a variety of prediction techniques, detailed
% in the following code. The first step is to prepare the data for
% prediction and to perform some exploratory analysis of the data.

%% Step 1: Data import
load workspace19aug2018.mat

yData = [];
responseY = [0,0];
V_B = 0.4;                   % flank wear threshold value
% specify interval length (in meters) between wear measurements
lengthIntervals = [ 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240];
timeIntervals = [0, lengthIntervals];
cuttingSpeed = cuttingParametersTable.cuttingSpeed(1);

% visualize the machining data
figure(1); clf reset;
for i = 1:size(machiningDataTable,1)
    timeIntervals = [0, lengthIntervals];
    cuttingSpeed = cuttingParametersTable.cuttingSpeed(i);
    % Obtain cutting timeframe by dividing the cutting distance by cutting
    % speed
    timeIntervals = (1000/cuttingSpeed)*timeIntervals;
    
    current = table2array(machiningDataTable(i,2:end));
    [cutoffVal, cutoffIndex] = max(current);
    % save cutoff vals and indeces in cv_ci array
    cv_ci(i,:) = [cutoffVal, cutoffIndex];
    responseY(i,:) = [cutoffVal, cutoffIndex];
    h= subplot(5,5,i);
    xrange = lengthIntervals(1:cutoffIndex);
    yrange = current(1:cutoffIndex);
    scatter(xrange,yrange,'b.')
    hold on;
    % plot a marker to indicate Vb
    plot([0,lengthIntervals],V_B*ones(1,size(lengthIntervals,2)+1),'r-.')
    xlabel('Length of cut (m)');
    ylabel('Wear value (mm)');
    title(h,['Experiment #' num2str(i)])
    xlim([0 max(xrange)+20]);ylim([0 cutoffVal]);
    hold off;
    
    % Interpolate the curve between v and 0.4 to find the stopping length
    % (and stopping time)
    
end

% Plot the maximum wear values given the cutting depth
figure(2); clf reset;

% extract the indices of the max elements
maxWearIntervals = responseY(:,2:end);

% create a new lengthIntervals array with the indices of these values
lengthIntervalVals = lengthIntervals(maxWearIntervals);
responseY(:,2:end) = lengthIntervalVals;

scatter(responseY(:,2),responseY(:,1))
ylabel('Max flank tool wear length (mm)');xlabel('Cutting Length (m)');

yData = responseY;
xData = table2array(cuttingParametersTable(:,:));

% combine the data into a single workspace variable
regressionData = [xData, yData];

%% Approach 1: Perform Regression Analysis on dataset

% SVM regression can be used to determine the correlation between predictor
% variables (i.e. cutting parameters) and repsonse variables (i.e.
% maxCuttingLength OR maxWearWidthValue). 

% We will use all avaible regression models to analyse the data

% Regression modelling of all machining parameters to predict max wear
% value returned the following models

predWearValues = {trainedMdl32, trainedMdl33, trainedMdl34, trainedMdl310, trainedMdl311, trainedMdl312};

% run some predictions to compare the models
predictorNames = {"column_1","column_2","column_3","column_4"};
% for i = 1:6
%     yPred(i) = predWearValues{i}.predictFcn(xSample,"VariableNames",predictorNames);
% end

%% Approach 2: Linear degradation model of milling tool wear

% We can estimate the model parameters using historical data regarding the
% health of an ensemble of similar components, such as multiple machine
% tools built to the same specifications, using the fit function.

% Model parameters can be specified when the model can be created based on
% knowledge of the component degradation process.

mdl = linearDegradationModel;

%% Approach 3: Deep Learning for health state classification
% Use image data to create a deep neural net classifier for ]
% "healthy" and "worn" tool states
imds = imageDatastore('ctd_experiment_files/');
imds.Labels = healthLabel;

% The original Excel file contains images in a specific pattern
% The files were renamed as exp_m_n_sp
% where m = 1:25 (experiment number)
% n is the cutting length (2,6,10,20,40,60,80,100,...,240)
% p = 01 or 02 depending on whether the image number was even or odd
% check you can open the renamed files

