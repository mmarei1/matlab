% use autoencoder to learn feature representation of the data as a subset
% of the entire feature space
%% Step 1: prepare the data for autoencoder by combining the training data
%  into an array
xtrain_data = horzcat(X_Train_n{:});
xtest_data = horzcat(X_Test_n{:});
g = gpuDevice;

%% Step 2: train the autoencoder with a set of configurable parameters to reconstruct the data
% try a feature dimension of 16
hiddenSize = 36;
reset(g);
autoenc1 = trainAutoencoder(xtrain_data, hiddenSize, ...
            'L2WeightRegularization', 0.004, ...
            'SparsityRegularization', 4, ...
            'SparsityProportion', 0.5, ...
            'MaxEpochs',1000,...
            'UseGPU',true);  
xReconstructed = predict(autoenc1, xtest_data);
%% Step 3: Measure the reconstruction loss of the autoencoder on the training and testing data
recloss = mse(xtest_data - xReconstructed)
%% Step 4: preprocess the training data using the autoencoder
feat1_train = encode(autoenc1,xtrain_data);
feat1_test = encode(autoenc1,xtest_data);
%% Step 5: re-transform the training and testing data from the encoded data
cellmap_train = [hiddenSize*ones(1,numel(sl_train));sl_train]'
cellmap_test = [hiddenSize*ones(1,numel(sl_test));sl_test]'
%%
X_Train_AE = mapMatToCellArray(feat1_train,size(feat1_train,1),sl_train');
X_Test_AE = mapMatToCellArray(feat1_test,size(feat1_test,1),sl_test');
% startIndex = 1;
% for i = 1:numel(sl_train)
%     endIndex = startIndex+sl_train(i)-1;
%     X_Test_AE{i} = mat2cell(feat1_test(:,startIndex:endIndex),cellmap_test(i,:));
%     startIndex = endIndex+1;
% end