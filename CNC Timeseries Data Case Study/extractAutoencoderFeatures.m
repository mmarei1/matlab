% deployAE = true;
% Extract features using autoencoder
% Use autoencoders to reduce the feature dimension of the data
% Note: this code expects the following inputs
% x_train: input training predictor sequences
% y_train: output training target values
% x_test: input testing predictor sequences
% y_test: output testing target values
% DO NOT RUN separately
if isempty(hiddenSize)
    hiddenSize = [48,36,28,16];
end

if deployAE == true
    fprintf("Feature Selection Approach: FSRNCA \n -----------------------------------------\n\n")
    
    % Prepare the sequences for feature extraction by concatenating all
    % predictor sequences into one cell array
    
    xtrain_data = horzcat(X_Train_n{:});
    xtest_data = horzcat(X_Test_n{:});
    g = gpuDevice;

    % Step 2: train the autoencoder with a set of configurable parameters to reconstruct the data
    % try a feature dimension of 16
    hiddenSize = [48,36,28,16];
    for i = 1:numel(hiddenSize)
        fprintf("Autoencoder Feature Reduction: %d to %d \n",size(xtrain_data,1),hiddenSize(i))
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
        X_AE_fspace(i) = size(feat1_train,1);
        X_Train_AE = mapMatToCellArray(feat1_train,X_AE_fspace(i),sl_train');
        X_Test_AE = mapMatToCellArray(feat1_test,size(feat1_test,1),sl_test');
        % Output the number of autoencoder features to be used
        disp(["Autoencoder Features Extracted:",size(X_Train_AE{1},1)])
        X_Train_AE_cell{i}=X_Train_AE;
        X_TEST_AE_cell{i}=X_Test_AE;
        %trainedAutoencoders{i} = struct("AutoencoderHiddenSize",hiddenSize(i),"AutoencoderModel",autoenc1{i},"ReconstructionLoss",recloss{i},"ReconstructedTrainingPredictors",X_Train_AE,"ReconstructedTestPredictors",X_Test_AE);
    end
    %save("Autoencoders.mat","trainedAutoencoders")
end