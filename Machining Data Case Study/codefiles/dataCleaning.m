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
cuttingSpeed = [cuttingParametersTable.cuttingSpeed(1)];

% visualize the machining data
figure(1); clf reset;
for i = 1:size(machiningDataTable,1)
    timeIntervals = [0, lengthIntervals];
    cuttingSpeed(i) = cuttingParametersTable.cuttingSpeed(i);
    % Obtain cutting timeframe by dividing the cutting distance by cutting
    % speed
    timeIntervals = (1000/cuttingSpeed(i))*timeIntervals;
    
    log_vc(i) = log(cuttingSpeed(i));
    
    current = table2array(machiningDataTable(i,2:end));
    [cutoffVal, cutoffIndex] = max(current);
    
    log_maxToolLife(i) = log((1000/cuttingSpeed(i))*cutoffIndex);
    
    
    % save cutoff vals and indeces in cv_ci array
    cv_ci(i,:) = [cutoffVal, cutoffIndex];
    responseY(i,:) = [cutoffVal, cutoffIndex];
    h = subplot(5,5,i);
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
    
    % generate a vector of predictor values for Vc, 
    xVector(i) = [cuttingSpeed(i), ];
end


% Plot the maximum wear values given the cutting depth
figure(2); clf reset;

% extract the indices of the max elements
maxWearIntervals = responseY(:,2:end);

% create a new lengthIntervals array with the indices of these values
lengthIntervalVals = lengthIntervals(maxWearIntervals);
responseY(:,2:end) = lengthIntervalVals;

scatter(responseY(:,2),responseY(:,1))
ylabel('Max flank tool wear width (mm)');xlabel('Cutting Length (m)');

yData = responseY;
xData = table2array(cuttingParametersTable(:,:));

% combine the data into a single workspace variable
regressionData = [xData, yData];


% plot log(vc) vs log(t)

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

%% Approach 2: Taylor's Tool Life equation for tool wear

% We can estimate the model parameters using historical data regarding the
% health of an ensemble of similar components, such as multiple machine
% tools built to the same specifications, using the fit function.

% Model parameters can be specified when the model can be created based on
% knowledge of the component degradation process.

% Taylor's tool life equation has the following form
% T = C_t/(V_c^x * f^y * a^z * H^d)
% where T is the tool life in minutes,
% Ct = Taylor constant,
% Vc = cutting speed in m/s,
% f = feed rate in mm/rev,
% a = depth of cut (mm)
% H = hardness in HRC - converted from the hardness in HB
% x, y, z, and d are coefficents to be estimated using the regression model

% Taking the natural log of both sides of the Taylor tool life equation:
% log(T) = 1*ln(Ct) + xln(V_c) + yln(f) + zln(a) + dln(H)

% Which leads to the generalised linear regression model
% Y - eps = 1*p0*ones(25,1) + p_1*x2 + p_3*x3 + p4*H*ones(25,1)

% where p(0) corresponds to the coefficient of C 
% p(1) corresponds to the exponent of V_c
% p(2) corresponds to the exponent of f
% p(3) corresponds to the exponent of a
% p(4) corresponds to the exponent of H

% Our model therefore has the following form
p1 = log(3500);
p2 = -0.9;
p3 = -0.6;
p4 = -0.2;
p5 = -0.025;

P = [p1*ones(25,1), p2*ones(25,1), p3*ones(25,1), p4*ones(25,1), p5*ones(25,1)]; 

y_est = @(X,p) p(:,1).*X(:,1) + p(:,2).*X(:,2) + p(:,3).*X(:,3) + p(:,4).*X(:,4) + p(:,5).*X(:,5);

my_fun = 'y ~ p(1)*x(1) + p(2)*log(x(2)) + p(3)*log(x(3)) + p(4)*log(x(4)) + p(5)*log(x(5))';

y_est2 = @(x,p)(p(1).^x(:,1)).*(x(:,2).^p(2)).*(x(:,3).^p(3)).*(x(:,4).^p(4)).*(x(:,5).^p(5));

% plot the matrix equations

% Y - eps = pX

p_hat = inv(X'*X)*X'*y

y_hat = X*p_hat

eps = y-y_hat

T_est = exp(y_hat)

mdl = fitnlm(X,y,y_est,p)

plotDiagnostics(mdl)

figure(5); clf reset
for i = 1:5
    indices(i,1:5) = i:5:(5-1)*5+i;
    hold on
    plot(log_vc(indices(i,:)),log_maxToolLife(indices(i,:)))
end

% create a piecewise interpolation function that performs the estimation
% over three stages in the wear process:
% t = 0:t_0#
