%% [src_reduced, tgt_reduced] = extractMinibatchFeatureOutputs(srcDataIn, tgtDataIn,dlnet, hiddenLayers, mbSize)
% Convert source and target domain data batches to feature representations
% through the layers specified in hiddenLayers
% Mohamed Marei, 2020

%%
function [src_reduced, tgt_reduced] = extractMinibatchFeatureOutputs(srcDataIn, tgtDataIn,dlnet, hiddenLayers, mbSize)
    % mbSize: number of observations
    % srcDataIn: Observation table, with data in the first column
    % tgtDataIn: Observation table, with target data in the first column
    % dlnet: the DLNetwork object used to extract the features
    % hiddenLayers: the hidden layers which are used to output the features
    n=mbSize;
        %layer = net1_MMD.Layers(end);
        % extract a minibatch of data from the source and target domain
        d_sz = size(cell2mat(table2array(layer.SourceData(1,1))));
        srcdata_sh = reshape(cell2mat(table2cell(layer.SourceData(1:n,1))),[d_sz n]);
        tgtdata_sh = reshape(cell2mat(table2cell(layer.SourceData(1:n,1))),[d_sz n]);        
        % transform the source and target mini-batches
        X_tgt = dlarray(single(srcdata_sh),'SSCB');
        X_src = dlarray(single(srcdata_sh),'SSCB');

        dlnet = layer.LayerGraph;
        dlnet = dlnetwork(dlnet);

        hlayers = layer.FeatureOutputs;
        X_tgt_frs = forward(dlnet,X_tgt,'Outputs',hlayers(2).Name);
        X_src_frs = forward(dlnet,X_src,'Outputs',hlayers(2).Name);

        src_reduced = single(X_src_frs);
        tgt_reduced = single(X_tgt_frs);
end