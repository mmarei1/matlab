% Script to extract model features from Fully Connected and Pooling Layers
%
% To use this script, the workspace should contain the following variables:
% Inputs: -Deep Learning model (net) and its name (passed to models{i})
%         -Augmented Image Datastores for the training and validation data,
%         which contain the images and associated predicted tool wear value
%         -regFlag: use "ensemble" to fit an ensemble of regressors; alternatively, 
%                   use an "svm" to fit a regression Support Vector Machine
%         
% The script produces the plots for the predicted values using the
% specified technique if the model contains pooling layers or fully
% connected layers. Otherwise, the outputs are empty
%
% Created by Mohamed Marei, 2019.
%%___________________________________________________________________________________

% regFlag = "ensemble";


fprintf("Using model %s as feature extractor: training %s regressors on extracted features \n",models{i},regFlag)

fclayers = findCNNFCLayers(net);
poolLayers = findCNNMaxPoolingLayers(net);

if ~isempty(fclayers)

    % final fc layer name

    l1 = net.Layers(fclayers(end-1)).Name;
    fprintf("FC Layer to use as feature extractor: %s \n",l1);
    fc_features_train = activations(net,augimdsTrain,l1,"OutputAs","rows");
    fc_features_train =  squeeze(mean(fc_features_train ,[1 2]))';

    fc_features_valid = activations(net,augimdsValidation, l1, "OutputAs","rows");
    fc_features_valid = squeeze(mean(fc_features_valid, [1 2]))';

    
    %% step 1: train ensemble using extracted FC features
    if strcmp(regFlag,"svm")
        regModel1 = fitrsvm(fc_features_train, ytrain);
    elseif strcmp(regFlag,"ensemble")
        regModel1 = fitrensemble(fc_features_train, ytrain);
    elseif strcmp(regFlag,"csvm")
        regModel1 = fitcsvm(fc_features_train, ytrain);
    else
        regModel1 = fitcecoc(fc_features_train, ytrain);
    end

    % predict outputs using regression ensemble
    ypred_re1 = predict(regModel1,fc_features_valid);
    ypred_re1_n = normalize(ypred_re1,'range',[0,1]);
    % Errors = predictions - targets
    errs_ens1 = ypred_re1 - yval;
    errs_ens1_n = ypred_re1_n - yval;
    % rmse of raw and normalized outputs
    rmse_ens1 = sqrt(mean(errs_ens1.^2))
    rmse_ens1_n = sqrt(mean(errs_ens1_n.^2))

    figure(1); clf reset;
    plot(1:numel(yval),yval,'bo');
    hold on;
    plot(1:numel(yval),ypred_re1_n,'ro');
    hold off;
    legend("Targets","Predictions")
    ylabel("Normalised Tool Wear Value (mm)")
    xlabel("Record Number")
    title(sprintf("Normalized Regression ensemble model output with respect to features extracted from final layer %s - RMSE = %3f",l1,rmse_ens1_n))
    %predictions_plotname = expNames{i}+"_extractedFeatures_"+p1+".png";
    predictions_plotname = expNames{i}+"_extractedFeatures_"+l1+"_"+regFlag+".png";
    saveas(gcf,predictions_plotname);

else
    fprintf("No FC layers found. Attempting to extract pooled features instead...\n")
    ypred_re1 = [];
end

%% step 2: train ensemble using pooled features
if ~isempty(poolLayers)
    p1 = net.Layers(poolLayers(end)).Name;
    % final pooling layer name
    fprintf("Pooling Layer to use as feature extractor: %s \n",p1);
    % Extract activations of final Pooling layer to use as regression ensemble
    % input
    
    pool1_features_train = activations(net,augimdsTrain,p1,"OutputAs","rows");
    pool1_features_valid = activations(net,augimdsValidation,p1,"OutputAs","rows");

    % train regression svm on pooled features
    if strcmp(regFlag,"svm")
        regModel2 = fitrsvm(pool1_features_train, ytrain);
    else
        regModel2 = fitrensemble(pool1_features_train, ytrain);
    end
    %regModel2 = fitrsvm(, ytrain);

    % predict outputs using regression ensemble
    ypred_re2 = predict(regModel2,pool1_features_valid);
    ypred_re2_n = normalize(ypred_re2,'range',[0,1]);
    % Errors = predictions - targets
    errs_ens2 = ypred_re2 - yval;
    errs_ens2_n = ypred_re2_n - yval;
    % rmse of raw and normalized outputs
    rmse_ens2 = sqrt(mean(errs_ens2.^2))
    rmse_ens2_n = sqrt(mean(errs_ens2_n.^2))
    
    %%
    figure(2); clf reset;
    plot(1:numel(yval),yval,'bo');
    hold on;
    plot(1:numel(yval),ypred_re2_n,'ro');
    hold off;
    legend("Targets","Predictions")
    ylabel("Normalised Tool Wear Value (mm)")
    xlabel("Record Number")
    title(sprintf("Regression ensemble model output with respect to features pooled from layer %s - RMSE = %3f",p1,rmse_ens2_n))
    predictions_plotname = expNames{i}+"_extractedFeatures_"+p1+"_"+regFlag+".png";
    %predictions_filename = expNames{i}+"_predictions.mat";
    saveas(gcf,predictions_plotname);
    
else
    fprintf("No pooling layers found. No more feature extraction can be employed...\n")
    ypred_re2 = [];
end