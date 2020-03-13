%% test definition of MMD loss function, ensuring a numerical output is reached
% inputs: 
    % model dlnetwork
    % model hidden layers
    % model source data (X_src)
    % model target data (X_tgt)
% the data is passed from within the transfer loss layer as tables. Extract
% the table values to use
        
        n=12;
        layer = net1_MMD.Layers(end);
        % extract a minibatch of data from the source and target domain
        d_sz = size(cell2mat(table2array(layer.SourceData(1,1))));
        srcdata_sh = reshape(cell2mat(table2cell(layer.SourceData(1:n,1))),[d_sz n]);
        tgtdata_sh = reshape(cell2mat(table2cell(layer.SourceData(1:n,1))),[d_sz n]);
        
        whos srcdata_sh tgtdata_sh
        
        %% transform the source and target mini-batches
        X_tgt = dlarray(single(srcdata_sh),'SSCB');
        X_src = dlarray(single(srcdata_sh),'SSCB');
%         end
%%
        dlnet = layer.LayerGraph;
        dlnet = dlnetwork(dlnet);
        %%
        hlayers = layer.FeatureOutputs;
        X_tgt_frs = forward(dlnet,X_tgt,'Outputs',hlayers(2).Name);
        X_src_frs = forward(dlnet,X_src,'Outputs',hlayers(2).Name);
        %%
%         X_src_g = stripdims(X_src_frs);
%         X_tgt_g = stripdims(X_tgt_frs);
%         
        X_src_f = cast(X_src_frs,'like',0.5);
        X_tgt_f = cast(X_tgt_frs,'like',0.5);
        
        % Calculate MMD in 2 steps:
        % Step 1: find optimal kernel combination between source and target
        % data representations within sigmas range specified as
        % sigmas = 2.^[-5:1;15];
        %%
        [bg] = opt_kernel_comb(X_src_f,X_tgt_f);
        % Step 2: calculate mmd from optimal kernel from source, target, optimal
        % kernels and corresponding kernel weights (which we assume to be 1)
        [sigma1,tmean,tvar,tcdf] = mmd_linear_combo(X_src_f,X_tgt_f,bg,1);
        % tmean is calculated in terms of MMD^2; loss is the square root of
        % MMD
        Loss = sqrt(tmean);
        % the gradient of the mmd loss w.r.t. learned parameters is found by 
        gradient = 1/2*sqrt(Loss);