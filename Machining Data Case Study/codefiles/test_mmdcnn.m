% run transfer learning experiment
load('cuttingToolImages_tables.mat');
% load('allFiles_Table.mat') % load all data
% load('cutting_tool_images_dataset_full.mat')
% source domain file directories on HPC and local systems
mainRoots_src = "/home/mareim/Downloads/cutting_tool_images/";
altRoots_src = 'C:\Users\Mareim\OneDrive - Coventry University\cutting_tool_images\';

% target domain data and datastore
mainRoots_tgt = "C:\Users\Mareim\OneDrive - Coventry University\Data\NewImages\Edge\";
altRoots_tgt = "/home/mareim/Downloads/NewImages/NewImages/Edge/"; 

fileRoots = allFiles_Table.FileName(:);
ds_src = imageDatastore(altRoots_src,'IncludeSubfolders',true,'FileExtensions','.jpg','Labels',allFiles_Table.Magnification(:),...
    'AlternateFileSystemRoots',[mainRoots_src,altRoots_src],'ReadFcn',@segmentAndCropImage_DS);
ds_tgt = imageDatastore(cdata_yest.filenames_nomanip,'IncludeSubfolders',true,'AlternateFileSystemRoots',...
[mainRoots_tgt,altRoots_tgt],'Labels',cdata_yest.ylabel(:),'ReadFc',@segmentAndCropImage_DS);

% test the output of the read function
testRead_src = read(ds_src);
testRead_tgt = read(ds_tgt);

whos testRead_src testRead_tgt
%%
% load target datastore
%load target_datastore.mat;

%load datastores_testMMD_CNN.mat
% view target and source datastores
%ds_src
%ds_tgt
%
%% Set up image preprocessing for both source and target datastore
% As we need to implement augmentation, we have to pass the function as a
% datastore read function as opposed to pre-transforming the image
%ds_src.ReadFcn = @segmentAndCropImage_DS;

%ds_tgt.ReadFcn = @segmentAndCropImage_DS;
rng('default');
%[ds_tgt_train,ds_tgt_val] = splitEachLabel(ds_tgt,0.8,0.2);
numTgt = 1:numel(cdata_yest.ylabel);
idxRatio = 0.8;
idxTrain = randperm(numTgt(end),ceil(idxRatio*numTgt(end)));
idxVal = setdiff(numTgt,idxTrain);
% take the indices 
ds_tgt_train = subset(ds_tgt,idxTrain);
ds_tgt_val = subset(ds_tgt,idxVal);
labels_train = cdata_yest.ylabel(idxTrain);
labels_val = cdata_yest.ylabel(idxVal);

ds_tgt_train_table = table(ds_tgt_train.Files,labels_train,'VariableNames',{'files','responses'});
ds_tgt_val_table = table(ds_tgt_val.Files,labels_val,'VariableNames',{'files','responses'});

%%
% create a subset of datastore equal in length to the number of files in
% training target datastore
ds_src = subset(ds_src,1:numel(ds_tgt_train.Files));
ds_src_table = table(ds_src.Files,ds_src.Labels(1:numel(ds_tgt_train.Files)),'VariableNames',{'files','responses'});

% We are sacrificing a great deal of speed to achieve this. But, we are
% still able to pre-process the data as intended
augmenter = imageDataAugmenter(...
    'RandRotation',[-30,30],...
    'RandXTranslation',[-30,30],....
    'RandYTranslation',[-30,30],...
    'RandScale',[0.8,1.1]);
inSize = [227 227];

%ds_src = readall
% ds_tgt_train_cell = readall(ds_tgt_train);
% ds_tgt_val_cell = readall(ds_tgt_val);
%
% ds_tgt_train_mat = cell2mat(ds_tgt_train_cell);
% ds_tgt_val_mat = cell2mat(ds_tgt_val_mat);
% tgt_train_labels = cdata_yest.ylabel(idxTrain);
% tgt_val_labels = cdata_yest.ylabel(idxVal);
%%
ds_src_aug = augmentedImageDatastore(inSize,ds_src_table,'DataAugmentation',augmenter,'OutputSizeMode','resize','ColorPreprocessing','gray2rgb')
ds_tgt_train_aug = augmentedImageDatastore(inSize,ds_tgt_train_table,'DataAugmentation',augmenter,'OutputSizeMode','resize','ColorPreprocessing','gray2rgb')
ds_tgt_val_aug = augmentedImageDatastore(inSize,ds_tgt_val_table,'DataAugmentation',augmenter,'OutputSizeMode','resize','ColorPreprocessing','gray2rgb')
% save datastores so we can re-use them in experiments
%save('datastores_testMMD_CNN.mat','ds_src_aug','ds_tgt_train_aug','ds_tgt_val_aug');
% load pretrained CNN
%%
net1 = alexnet;
% pass the datastores as the source and target training sources
net1 = layerGraph(net1.Layers);
%%
% ds_src_train_tf = transform(ds_src,@(X)imresize(X,[227 227]));
% ds_tgt_train_tf = transform(ds_tgt_train,@(X)imresize(X,[227 227]));
% ds_tgt_val_tf = transform(ds_tgt_val,@(X)imresize(X,[227 227]));
net1_MMD = createMMD_CNN(net1,"mse-mmd",[0.9, 0.1],ds_src_aug,ds_tgt_train_aug);
net1_MSE = createMMD_CNN(net1,"mse",[1,0],{},{});
save('net_MMD.mat','net1_MMD');
%% Specify network training options
options = trainingOptions('adam','MaxEpochs',10,'MiniBatchSize',12,'Verbose',true','plots','training-progress','InitialLearnRate',4e-5);
%Train Network
%% Check the implementation of the MMD layer

checkLayer(net1_MMD.Layers(end),viSize)

%%
[net1_MMD_trained,info] = trainNetwork(ds_tgt_train_aug,net1_MMD,options);
% Predict network outputs for the validation dataset
YPred = predict(net1_MMD_trained,ds_tgt_val_aug);
%%
% outputs
outputs = table(labels_val,YPred,'VariableNames',{'Targets','Predictions'});
%Plot the results
figure(1);clf reset;
scatter(1:numel(outputs.Targets),outputs.Targets(:),'b');
hold on;
scatter(1:numel(outputs.Predictions),outputs.Predictions(:),'r');
title('Targets vs Predictions with AlexNet_MMD');
save('validation_results_alexnet.mat','outputs');

