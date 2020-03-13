function [confusion1, confusino2, stats] = predictUsingPretrainedNet(net, datastorePath, dataSplit, augmentationOptions)
    % Function to generate predictions on the tool images condition monitoring
    % dataset using pretrained deep learning models
    %
    % function inputs:
    %
    % net - a handle to the initialized deep learning network (e.g. AlexNet, VGGNet, etc)
    %
    % datastorePath - the path to the image datastore
    %
    % datasplit - a vector containing n <=3 values,  specifying the
    % proportion of the data for training, testing and validation. For
    % example, use [0.5 0.3 0.2] if you want to use 50% of the data for
    % training, 30% of the data for testing and 20% for validation.
    % 
    % augmentationOptions - a vector of n <= 3 values that specify
    % additional data augmentation values (default is 'none')
    % 
    % function outputs:
    % 
    % confusion matrix for feature extractor 
    % 
    % confusion matrix for transfer learning
    % 
    % accuracy statistics on the dataset
    % 
    % Example: Run an instance of AlexNet for feature extraction and
    % transfer learning
    %   net = alexnet;
    %   ffpath = C:\...\NewImages\Edge'
    %   data = imageDatastore(ffpath,'IncludeSubfolders',true,'LabelSource','foldernames');
    %   [c1, c2, stats] = predictUsingPretrainedNet(net, data, dataSplit, augOptions);
    
    help
end