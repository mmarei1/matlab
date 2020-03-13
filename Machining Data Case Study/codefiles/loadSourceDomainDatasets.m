%% Load data as table
%addpath(genpath(pwd))
function [data_src_train, data_src_val] = loadSourceDomainDatasets(sources,selectionMethod,desiredSize)
    load('allFiles_Table.mat');
    % selection criterion of files: Magnification = 100;
    tf1 = allFiles_Table.Magnification == 100;
    % Create first source domain datastore based on subset of images defined by tf1;
    % Labels assigned based on variant
    totalFiles = numel(find(tf1));
    
    rng('default');
    if strcmp(selectionMethod,'random')
        idxSelection = randperm(totalFiles,desiredSize);
    else
        idxSelection = 1:desiredSize;
    end
    altRoots = "C:\Users\Mareim\OneDrive - Coventry University\cutting_tool_images";
    mainRoots = "/home/mareim/Downloads/cutting_tool_images";
    source_ds = imageDatastore(allFiles_Table.FileName(tf1),'IncludeSubfolders',true,'AlternateFileSystemRoots',[mainRoots,altRoots]);
    if sources ==1
        clabels = categorical(allFiles_Table.Variant(tf1),'Ordinal',0);
    elseif sources ==2
        clabels = categorical(allFiles_Table.CuttingDistance(tf1),'Ordinal',0);
        %clabels_s2 = categorical(allFiles_Table.CuttingDistance(tf1),'Ordinal',0);
    elseif sources ==3  
        clabelNames = {'x0','x1','x2','x3','x4','x5','x6','xF'};
        [clabels,~] = discretize(allFiles_Table.CuttingDistance(tf1),8,'categorical',clabelNames);
    else
        clabels = "none";
    end
    source_ds.Labels = clabels;
    %source_ds.ReadFcn = @segmentAndCropImage_DS;
    source_ds_subset = subset(source_ds,idxSelection);
    [data_src_train,data_src_val] = splitEachLabel(source_ds_subset,0.7,0.3);
end

