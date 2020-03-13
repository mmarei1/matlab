%% Import the data and pre-process it for Deep Learning model training
load('modelData.mat');
%% autoencoder parameters
hiddenSize = 36;
l2weightReg = 0.004;
sparsityReg = 6;
sparsityProp = 0.8;
maxEpochs = 1000;
deployAE = false;
% Transpose each cell array such that the feature dimension is the number
% of rows
X_Train =  cellfun(@transpose,XData_train_cell,'UniformOutput',false);
X_Test = cellfun(@transpose,XData_test_cell,'UniformOutput',false);
Y_Train = cellfun(@transpose,YData_train_cell,'UniformOutput',false);
Y_Test = cellfun(@transpose,YData_test_cell,'UniformOutput',false);

% calculate the mean and standard deviation of each feature
[mu,sigma] = normalizeFeatures(X_Train);
%%
% calculate the min and max of Y_train_n
[Y_Train_n, ymin, ymax, Y_Train_u] = normalizeOutputRange(Y_Train,0,1);
[Y_Test_n, ~, ~, ~] = normalizeOutputRange(Y_Test,0,1);

% YMIN = repmat({ymin},1,12);
% YMAX = repmat({ymax},1,12);
% ydiff = repmat({ymax-ymin},1,12);

%Y_Test_n = cellfun(@(A,B,C) (A(:)-B(:))./(C), Y_Test,YMIN,ydiff,'UniformOutput',0);

%%
% create new X_Train adjusted by the mean and standard deviation of the
% features
% additionally, add the output from y(t-1) as a feature of x
alpha = 0.8;
windowSize = 1;
for i = 1:numel(X_Train)
    tmpX = cell2mat(X_Train{i});
    tmpX = (tmpX - mu)./sigma;
    tmpY = cell2mat(Y_Train{i});
    sl_train(i) = numel(tmpY);
    Y_Train_n{i} = tmpY;
    Y_train_lag1 = [alpha*tmpY(windowSize),tmpY(windowSize:end-1)];
    Y_train_lag2 = [alpha*Y_train_lag1(1),Y_train_lag1(1:end-1)];
    % add the y(t-1) lagged responses to the input vector as inputs;
    % assume the last value is repeated twice
    X_Train_n{i} = [tmpX;Y_train_lag1;Y_train_lag2];
end
 
for i = 1:numel(X_Test)
    tmpX = cell2mat(X_Test{i});
    tmpX = (tmpX - mu)./sigma;
    tmpY = cell2mat(Y_Test{i});
    sl_test(i) = numel(tmpY);
    Y_Test_n{i} = tmpY;
    Y_test_lag = [alpha*tmpY(windowSize),tmpY(windowSize:end-1)];
    Y_test_lag2 = [alpha*Y_test_lag(1),Y_test_lag(1:end-1)];
    X_Test_n{i} = [tmpX;Y_test_lag;Y_test_lag2];
end

X_Train_n = X_Train_n';
X_Test_n = X_Test_n';
Y_Train_n = Y_Train_n';
Y_Test_n = Y_Test_n';

size(X_Train_n)
size(Y_Train_n)
size(X_Test_n)
size(Y_Test_n)

disp(["Number of features: ",size(X_Train_n{1},1)]);
%
% use autoencoder to learn feature representation of the data as a subset
% of the entire feature space
%% Step 1: prepare the data for autoencoder by combining the training data
%  into an array
deployAE = true;

if deployAE == true
    xtrain_data = horzcat(X_Train_n{:});
    xtest_data = horzcat(X_Test_n{:});
    g = gpuDevice;

    % Step 2: train the autoencoder with a set of configurable parameters to reconstruct the data
    % try a feature dimension of 16
    hiddenSize = [48,36,28,16];
    for i = 1:numel(hiddenSize)
    reset(g);
    autoenc1{i} = trainAutoencoder(xtrain_data, hiddenSize(i), ...
                'L2WeightRegularization', l2weightReg, ...
                'SparsityRegularization', sparsityReg, ...
                'SparsityProportion', sparsityProp, ...
                'MaxEpochs',2000,...
                'UseGPU',true);  
    xReconstructed = predict(autoenc1{i}, xtest_data);
    % Step 3: Measure the reconstruction loss of the autoencoder on the training and testing data
    recloss{i} = mse(xtest_data - xReconstructed)
    % Step 4: preprocess the training data using the autoencoder
    feat1_train = encode(autoenc1{i},xtrain_data);
    feat1_test = encode(autoenc1{i},xtest_data);
    % Step 5: re-transform the training and testing data from the encoded data
    X_Train_AE = mapMatToCellArray(feat1_train,size(feat1_train,1),sl_train');
    X_Test_AE = mapMatToCellArray(feat1_test,size(feat1_test,1),sl_test');
    disp(["Autoencoder Features Extracted:",size(X_Train_AE{1},1)])
    X_Train_AE_cell{i}=X_Train_AE;
    X_TEST_AE_cell{i}=X_Test_AE;
    %trainedAutoencoders{i} = struct("AutoencoderHiddenSize",hiddenSize(i),"AutoencoderModel",autoenc1{i},"ReconstructionLoss",recloss{i},"ReconstructedTrainingPredictors",X_Train_AE,"ReconstructedTestPredictors",X_Test_AE);
    
    end
    %save("Autoencoders.mat","trainedAutoencoders")
end