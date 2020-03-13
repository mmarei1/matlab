% run experiments using custom training loop
% load files containing target dataset
load('cutting_tool_images_dataset_full.mat');


target_ds = cdata_yest.filename(:);
target_Ydata = cdata_yest.ylabel(:);
% divide target dataset into training and validation sets
rng('default');
splitRatio = 0.7;
idx_train = randperm(ceil(0.7*numel(target_Ydata)));
idx_val = setdiff(1:numel(target_Ydata),idx_train);
% select the filenames with idx_train
ds_train = target_ds(idx_train);
Y_train = target_Ydata(idx_train);
% select the filenames with idx_val
ds_val = target_ds(idx_val);
Y_val = target_Ydata(idx_val);

% create target training and validation datastores
train_ds_tgt = imageDatastore(ds_train,'LabelSource','none','ReadFcn',@segmentAndCropImage_DS);
val_ds_tgt = imageDatastore(ds_val,'LabelSource','none','ReadFcn',@segmentAndCropImage_DS);
%%
XTrain_tgt = train_ds_tgt;
YTrain_tgt = Y_train;
dsSize = numel(Y_train);
% try the first source dataset
%[ds_train_src,ds_val_src] = loadSourceDomainDatasets(1,dsSize,"random");
%% Prepare source pre-training experiments
sourceExperiments = [3];
models = {'alexnet'};
%,'resnet18','resnet50','resnet101','squeezenet','inceptionv3'};
losses = {'MMD'};

imaug = imageDataAugmenter('RandRotation',[-20,20],'RandXTranslation',[-20,20],'RandYTranslation',[-20,20],'RandScale',[0.9,1.2]);

% this will load training options
for i=1:numel(sourceExperiments)
    for n = 1:numel(models)
        % load each source domain dataset for pre-training
        [src_train, src_val] = loadSourceDomainDatasets(3,"random",dsSize);

        % load the model with the correct source labels as prediction outputs
        % objective of the model is cross-entropy loss

        src_labels = unique(src_train.Labels);
        [net,expName] = loadPretrainedCNN("alexnet",src_labels,"cross",0,10,[],[],[]);
        nets{n} = net;
        expNames{n} = expName;
        imInputSize = nets{n}.Layers(1).InputSize;
        % augment the image datastore with the augmentation options desired
        imdsTrain = augmentedImageDatastore(imInputSize,src_train,'DataAugmentation',imaug,'OutputSizeMode','Resize')
        % validation dataset augmentation is not desired
        imdsVal = augmentedImageDatastore(imInputSize,src_val,'DataAugmentation','none','OutputSizeMode','Resize')
       
        options_pt = loadDefaultTrainingOptions("pt",imdsVal);

        %options_pt.ValidationData = src_val;
        
        % fine tune the network
        [net_pt,info] = trainNetwork(imdsTrain,nets{n},options_pt);
        % 
        YActual = src_val.Labels;
        %% Validation
        YPred = classify(net_pt,imdsVal);
        tf = YPred == YActual;

        correctPredictions = numel(find(tf));

        validationAccuracy = correctPredictions/numel(YActual)
        
        fprintf('Validation Accuracy for model after pre-training: %2.2f% \n',100*validationAccuracy)
        fprintf('Source task: %d \n',sourceExperiments(i))
        fprintf('Class labels: \n')
        unique(YPred)
        %%
        figure; clf reset;
        plotconfusion(YActual,YPred)
        %title(sprintf("Confusion chart for %s", expNames{i}));
        
        trained_nets{i} = net_pt;
        %%
        % attempt to create CNN-LSTM from pre-trained CNN
        
        %lgraph1 = layerGraph();
        %try
%         clear lgraph tempLayers;
%         lgraph = layerGraph(net_pt.Layers);
%         
%         [fclayer,classLayer,smLayer] = findBottomLayersToReplace(lgraph);
%         lgraph = removeLayers(lgraph,lgraph.Layers(1).Name);
%         tempLayers1 = ...
%             [sequenceInputLayer(imInputSize,'Name','SequenceInputLayer','Normalization','zerocenter'),...
%             sequenceFoldingLayer('Name','seqfold'),...
%             ];   
%         lgraph = removeLayers(lgraph,{classLayer.Name,smLayer.Name});
%         lgraph = addLayers(lgraph,tempLayers1);
%         %%
%         %tempLayers = [];
%         tempLayers2 = ...
%             [sequenceUnfoldingLayer('Name','sequnfold'),...
%             flattenLayer('Name','flattenLayer'),...
%             lstmLayer(128,'Name','LSTM1'),...
%             fullyConnectedLayer(1,'Name','LSTM_fc'),...
%             sigmoidActivationLayer(1,'Sigmoid_final'),...
%             regressionLayer('Name','RegressionOut')
%         ];
%         lgraph = addLayers(lgraph,tempLayers2);
%     % connect everything up
%         lgraph = connectLayers(lgraph,"seqfold/out","conv1");
%         lgraph = connectLayers(lgraph,"seqfold/miniBatchSize","sequnfold/miniBatchSize");
%         lgraph = connectLayers(lgraph,fclayer.Name,"sequnfold/in");
%       Function to add additional LSTM layers to the CNN
%       Input arguments: CNN base
%       density (number of LSTM hidden units - 128,256, etc)
%       depth (number of LSTM units >=1) 
%       weights initializer (Glorot)
        lgraph = createCLSTM(net_pt,256,2,'glorot');
        analyzeNetwork(lgraph)
        %%
         
        XTrain = cdata_yest(idx_train,{'filename','ylabel'});
        XVal = cdata_yest(idx_val,{'filename','ylabel'});
        
        augimdsTrain = augmentedImageDatastore(imInputSize(1:2),XTrain, ...
        'DataAugmentation',imaug);
        imdsVal = augmentedImageDatastore(imInputSize(1:2),XVal,'DataAugmentation','none','OutputSizeMode','Resize')
        
        optionsLSTM = trainingOptions('adam',...
            'MaxEpochs',20,...
            'MiniBatchSize',16,...
            'plots','training-progress',...
             'InitialLearnRate',2e-4,...
             'LearnRateDropFactor',0.9,...
             'LearnRateDropPeriod',10,...
             'ValidationData',imdsVal,...
             'ValidationFrequency',10,...
             'SequenceLength','shortest',...
             'shuffle','never');
         
         [lstm_cnn_trained, info] = trainNetwork(augimdsTrain,lgraph,optionsLSTM)
        %catch MExc
            %mtext = getreport(MExc);
            %warning('Failed to create LSTM-CNN; Reason: ',mtext)
        %end
        YPred_CLSTM = predict(lstm_cnn,Y_tgt);
        %%
        % convert trained net to layerGraph
        dlnet_noreg = layerGraph(net_pt.Layers(1:end-3))
        
        % convert the entire network less the output layers (i.e. softmax
        % and regression layers)
        dlnet = dlnetwork(dlnet_noreg);
        %dlnet = fullyconnect(dlnet,)
        %[] = runCustomTrain();
        options_ft = loadDefaultTrainingOptions("ft",'none')
        XTrain_src = src_train;
        YTrain_src = src_train.Labels;
        [trainedNet,info] = runCustomTrain(dlnet,options_ft,XTrain_tgt, YTrain_tgt, XTrain_src, YTrain_src);
        
        YPred_
    end    
end