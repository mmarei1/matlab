%% Summary
% This code tests the use of pretrained deep learning networks for feature
% extraction and transfer learning, tested on the data we have. The first
% approach computes the activations of the last layers of AlexNet and uses
% that layer as a feature extractor for our data. The second approach
% applies Transfer Learning to our problem, where a tweaked pretrained AlexNet is
% used with modified softmax and classification layers to predict on our
% data.
% 
% Possible nets: vgg16 vgg19 resent18 resnet50 resnet101
% Uncomment the corresponding line to select the number of "top" layers
% which will be used for Transfer Learning. Those are (usually) all the
% layers before 'fc7'. After these layers, we implement our own "bottom
% layers" which contain the classification structure relevant to our
% problem.
% 
pretrained_net = vgg16;
topLayers = pretrained_net.Layers(1:37);
% % AlexNet
% topLayers = pretrained_net.Layers(1:20);
% % VGG16
% topLayers = pretrained_net.Layers(1:37);
% % vgg19
% topLayers = pretrained_net.Layers(1:43);
% % resent18
% topLayers = pretrained_net.Layers(1:37);
% % resent50
% topLayers = pretrained_net.Layers(1:37);
% % resent101
% topLayers = pretrained_net.Layers(1:37);
% % *resent18 - note the bases have different structures
% topLayers = pretrained_net.Layers(1:37);
% % *resneet101 - note the bases have different structures
% topLayers = pretrained_net.Layers(1:37);
% topLayers = pretrained_net.Layers(1:37);

%% Approach 1: Feature extraction using the fully connected layer weights from AlexNet

% Step 1 - load images to a datastore object
ptmImds = imageDatastore('C:\Users\mareim\Documents\MATLAB\NewSandbox\Machining Data Case Study\NewImages\Edge','IncludeSubfolders',true,'LabelSource','foldernames')
classLabels = countEachLabel(ptmImds)
% Step 2 - split data into training, testing and validation datasets
[imdsTrain, imdsTest, imdsValidation] = splitEachLabel(ptmImds,0.5 ,0.25, 0.25 );
%[imdsTrain, imdsTest] = splitEachLabel(ptmImds,0.6 ,0.4);
inputSize = pretrained_net.Layers(1).InputSize;
imageAugmentation = imageDataAugmenter('RandRotation',[-20,20],'RandXTranslation',[-100, 100],'RandYTranslation',[-100, 100]);
% Step 3 - augment the datasets individually so they can fit in inputSize
imdsTrainAug = augmentedImageDatastore(inputSize, imdsTrain)
imdsTestAug = augmentedImageDatastore(inputSize, imdsTest)
imdsValnAug = augmentedImageDatastore(inputSize, imdsValidation)

% Step 4 - designate AlexNet weights to be used for the prediction model
layersArray = 'fc7'

% Step 5 - calculate the activations of the training and testing datasets
% on this layer (this is the only computation we need)
fTrain = activations(pretrained_net,imdsTrainAug,layersArray,'OutputAs','rows');
fTest = activations(pretrained_net,imdsTestAug,layersArray,'OutputAs','rows');

% Step 5.5 - fine-tuning

% Step 6 - Extract the class labels from training and testing data
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
YVal = imdsValidation.Labels;

% Step 7 - predict the classes using a SVM model (fitecoc)
classifier = fitcecoc(fTrain, YTrain,'Cost',[0 1; 30 0]);

% Step 8 - classify test images on the trained SVM
YPred = predict(classifier, fTest);

accuracy = mean(YPred == YTest);
cfmat = confusionmat(YPred,YTest);
plotconfusion(YPred,YTest)

%% Now let's try something new with AlexNet. 

% modify AlexNet's last three layers to better suit the classification
% task
% for AlexNet, we are interested in the top 20 layers
% for VGG16, we are interested int he top 35 layers
% ...
frankenstein = [topLayers; 
    dropoutLayer('Name','drop7'); 
    fullyConnectedLayer(2, 'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fc8');
    softmaxLayer('Name','softmax'); 
    classificationLayer('Name','classification')
    ];
% set training options dependent on problem spec and size of data
frank_options = trainingOptions('sgdm', ...
    'MiniBatchSize',30, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValnAug, ...
    'ValidationFrequency',5, ...
    'Verbose',true, ...
    'Plots','training-progress');
% retrain and test the hybrid network
netTransfer = trainNetwork(imdsTrainAug, frankenstein, frank_options);
[YPred,scores] = classify(netTransfer,imdsValnAug);
% attempt a different classifier here
discriminant_model = fitcdiscr(netTransfer,imdsValnAug,'Cost',[0 1; 0 30]);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
figure(2); clf reset
cfmat = confusionmat(YPred,YValidation);
plotconfusion(YPred,YTest)

