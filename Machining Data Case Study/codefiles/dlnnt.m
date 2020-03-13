%% Approach 3: Deep Learning for health state classification
% Use image data to create a deep neural net classifier for ]
% "healthy" and "worn" tool states
% load workspace20180905
imds = imageDatastore('C:\Users\mareim\Documents\MATLAB\NewSandbox\Machining Data Case Study\NewImages\Edge','IncludeSubfolders',true,'LabelSource','foldernames')

% Can the network classify the wear states correctly given just the
% preprocessed image (with no determination of whether the image is of a
% healthy or worn cylinder or edge)?

% Split the data into training, testing and validation datasets
[imdsTrain, imdsTest, imdsValidation] = splitEachLabel(imds,0.6 ,0.2,0.2 );
outputSize = [320 240];
% imdsTrainAug = augmentedImageDatastore(outputSize, imdsTrain, 'ColorPreprocessing','rgb2gray');
% imdsTestAug = augmentedImageDatastore(outputSize, imdsTest, 'ColorPreprocessing','rgb2gray');
% imdsValnAug = augmentedImageDatastore(outputSize, imdsValidation, 'ColorPreprocessing','rgb2gray');

imdsTrainAug = augmentedImageDatastore(outputSize, imdsTrain);
imdsTestAug = augmentedImageDatastore(outputSize, imdsTest);
imdsValnAug = augmentedImageDatastore(outputSize, imdsValidation);

% set the training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...,
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.0725,...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValnAug, ...
    'ValidationFrequency',1, ...
    'Verbose',false, ...
    'MiniBatchSize',100,...
    'Plots','none', ...
    "ExecutionEnvironment","parallel");

% Network Structure
% things to try:
% 1- modify filter number based on the nature of the image
% 2- modify pooling layer parameters
% 3- tune parameters based on picture "data"
% 4- data augmentation
layers = [
    imageInputLayer([outputSize 3], 'Name', 'input')
    
    convolution2dLayer(5,100,"Padding", "same", "Stride", [4 4] ,'Name', 'C1')
    
    maxPooling2dLayer([4 3], "Stride", 2,'Name', 'MP1')
    
    convolution2dLayer(5,160, "Padding", "same",'Name', 'C2')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'RL1')
    
    maxPooling2dLayer([4 3], "Stride", 2,'Name', 'MP2')
    
    convolution2dLayer(5,260,"Padding", "same",'Name', 'CL3')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'RL2')

    fullyConnectedLayer(2,'Name', 'FC1')
    softmaxLayer('Name', 'SM1')
    classificationLayer('Name', 'Classification')
    ];

lg = layerGraph(layers)

% Network Structure with a different network
layers2 = [
    imageInputLayer([outputSize 1])
    
    convolution2dLayer(7,25,"Padding", "same")
    
    averagePooling2dLayer(2, "Stride", 2)
    
    convolution2dLayer(5,50, "Padding", "same")
    batchNormalizationLayer
    leakyReluLayer
    
    averagePooling2dLayer(2, "Stride", 2)
    
    convolution2dLayer(5,100,"Padding", "same")
    batchNormalizationLayer
    leakyReluLayer
   
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

% net = trainNetwork(imdsTrainAug, layers, options);

% YPred = classify(net, imdsTestAug);
% yc1 = confusionmat(imdsTest.Labels, YPred)

%net2 = trainNetwork(imdsTrain, layers, options);

%YPred = classify(net2, imdsTest);
%yc2 = confusionmat(imdsTest.Labels, YPred)

% Trim AlexNet and combine it with other layer and modified learning
% parameters
frankenstein = [clanet_mod; 
    dropoutLayer('Name','drop7'); 
    fullyConnectedLayer(2, 'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fc8');
    softmaxLayer('Name','softmax'); 
    classificationLayer('Name','classification')
    ];

frank_options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

