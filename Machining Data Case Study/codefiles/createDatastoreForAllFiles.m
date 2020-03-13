% Pre-training the network layers on datastore of all images to improve
% feature transferability

%% Step 1: Use pre-trained models to learn features relevant to existing dataset by learning to classify healthy and worn tools
% from all provided images
% This step adapts the network to fine-tune the weights of the networks so
% that they learn relevant features to the specified problem
% Use the softmax as the loss metric for this binary classification
% problem
wd = "/home/mareim/Downloads/cutting_tool_images"
ctds_AllFiles = imageDatastore(wd,"IncludeSubfolders",1);
FileName = ctds_AllFiles.Files;
for i = 1:length(FileName)
    % function to parse string filename into ExpLabel, DistLabel, and
    % WearState
    [experiment,cutd,magnif,variant] = parseLabelsFromFilenames(FileName{i},wd);
    Experiment(i) = experiment;
    CuttingDistance(i) = cutd;
    Magnification(i) = magnif;
    Variant(i) = variant;
    idx(i) = i;
end
allFiles_Table = table(idx',FileName,Experiment', CuttingDistance',Magnification',Variant',...
'VariableNames',{'Index','FileName','Experiment','CuttingDistance','Magnification','Variant'});

%% Step 1.5 Select only the relevant images based on a magnification value of 100 and the variant is not 3;
% display a sample of the selected images
tf_targetDomain = (allFiles_Table.Magnification == 100 & (allFiles_Table.Variant ~= 3));
selected_targetDomain = allFiles_Table(tf_targetDomain,:);
% figure(1); clf reset;
% h1 = histogram(selected_targetDomain.CuttingDistance,"NumBins",...
%     numel(unique(selected_targetDomain.CuttingDistance)));
%% Create source dataset 2
tf
source2_ds = imageDatastore()
% split the labels 0.7 for training, 0.15 for validation and 0.15 for
% testing
% discretize the cutting distance labels
clabels = {'x0','x1','x2','x3','x4','x5','x6','xF'};

[bins_d,edges_d] = discretize(selected_targetDomain.CuttingDistance(:),8,'categorical',clabels);
ds_sz = height(selected_targetDomain)
% change to non-ordinal array
bins_d = categorical(bins_d,'Ordinal',0);
%bins_char = char(bins_d);
imds_selected = imageDatastore(selected_targetDomain.FileName,'LabelSource','none');
% Now we have selected the appropriate data, we can load the pre-trained
% versions of the nets so we can fine-tune the network layers on them
imds_selected.Labels = bins_d;
[ds_train, ds_val] = splitEachLabel(imds_selected,0.7,0.3);

% Weights Matrix set to have low values for each of the more common
% cutting distances and a high value for the less common cutting distances

%weightsFCFinal = diag([1,1,1,10,10,50,100,100],0);
%%
% Try first with AlexNet
%clear inputSize;
%models = ["resnet101"];
opts_ft = trainingOptions('adam', ...
        'MiniBatchSize',16, ...
        'MaxEpochs',15, ...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropPeriod',10,...
        'LearnRateDropFactor',0.9,...
        'InitialLearnRate',2e-5, ...
        'Shuffle','never', ...
        'ValidationData',ds_val, ...
        'ValidationFrequency',10, ...
        'ValidationPatience',Inf, ...
        'Verbose',true, ...
        'Plots','training-progress',...
        'ExecutionEnvironment',exEn);
%     %%
% % todo: add additional
% for i = 1:numel(models)
%     % TO DO: fix classification labels
%     [tmpNet, TmpExpNames] = loadPretrainedCNN(models(i),clabels,"cross",0);
%     expNames{i} = TmpExpNames+"_finetuning";
%     inputSize{i} = tmpNet.Layers(1).InputSize;
%     nets{i} = tmpNet;
%     [tmpNet_ft,valAcc_ft] = finetuneModel(tmpNet, ds_train, ds_val)
% end