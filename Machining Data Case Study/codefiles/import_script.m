clear; clc; close all;
filenames = ls;
filenames = string(filenames);
fn = sort(filenames(contains(filenames,'results')));
fn = erase(fn," ");
% load each result into a table
%bestTT = ["a","b","c","d","e","f"];
%bestMAE = ["a","b","c","d","e","f"];
%bestAcc = ["a","b","c","d","e","f"];
variants = [" (ADAM)", " (SGDM)", " (RMSPROP)"];
colormap lines
% preset the default figure font size
set(0,'DefaultAxesFontSize',16);
figure(1); clf reset
subplot(2,3,1)

for i= 1:size(fn)
    subplot(2,3,i)
    networkResults = load(fn(i));
    rt1 = networkResults;
    % retrieve network names;
    tstr1 = rt1.results(:,1).NetworkName;
    tstr2 = rt1.results(:,2).NetworkName;
    tstr3 = rt1.results(:,3).NetworkName;
    % retrieve network name substring
    NetworkName = fn(i);
    NetworkName = erase(NetworkName,"results-");
    NetworkName = erase(NetworkName,".mat");
    if NetworkName ~= "googlenet"
    titlestring = "Model Name: " + NetworkName;
    modelComparisons = [rt1.results.validationData,rt1.results(:,1).Predictions,rt1.results(:,2).Predictions,rt1.results(:,3).Predictions];
    bw2 = 0.1;
    bw = 0.05;
    h1 = histogram(modelComparisons(:,1),'BinWidth',bw2);
    hold on
    h2 = histogram(modelComparisons(:,4),'BinWidth',bw);
    h3 = histogram(modelComparisons(:,5),'BinWidth',bw);
    h4 = histogram(modelComparisons(:,6),'BinWidth',bw);
     
    xlabel('Normalised Wear Value');
    ylabel('Count');
    legend('Target Data','Predictions (ADAM)','Predictions (SGDM)','Predictions (RMSPROP)');
    title(titlestring)
    hold off;
    %%%%%%%%%%%%
    figure(21);subplot(2,3,i);binedge=-0.2:bw2:1;
    colormap jet;
    h2 = histogram(modelComparisons(:,4),binedge); hold on;
    h3 = histogram(modelComparisons(:,5),binedge);
    h4 = histogram(modelComparisons(:,6),binedge);
    h1 = histogram(modelComparisons(:,1),binedge,'FaceAlpha',0.15);
    hold off;
    figure(22);subplot(2,3,i)
    colormap lines;
    binedge=binedge+bw2/6;
    bar(binedge(1:end-1),[h2.Values],1/3);
    hold on;bar(binedge(1:end-1)+bw2/3,[h3.Values],1/3);
    bar(binedge(1:end-1)+2*bw2/3,[h4.Values],1/3');
    histedges = 0:0.1:1;
    h1 = histogram(modelComparisons(:,1),histedges,'FaceAlpha',0.15,'LineWidth',1);
    xlabel('Normalised Wear Value');
    ylabel('Count');
    legend('Predictions (ADAM)','Predictions (SGDM)','Predictions (RMSPROP)','Target Data','FontSize',10);
    title(titlestring)
    hold off;
%     figure(23);subplot(2,3,i);
%     colormap jet;
%     plot(binedge(1:end-1),[h1.Values],'g');hold on;
%     plot(binedge(1:end-1),[h2.Values],'r');
%     plot(binedge(1:end-1),[h3.Values],'y');
%     plot(binedge(1:end-1),[h4.Values],'b');
%     xlabel('Normalised Wear Value');
%     ylabel('Count');
%     legend('Target Data','Predictions (ADAM)','Predictions (SGDM)','Predictions (RMSPROP)');
%     title(titlestring)
%     hold off;
    
    %%%%%%%%%%%%%%% Retrieve comparisons
    NetLabels(i) = string(NetworkName);
    % get the minimum training times for all model variants
    allTTs(i,:) = [...
        networkResults.results(:,1).Performance.TrainingTime, ...
        networkResults.results(:,2).Performance.TrainingTime, ...
        networkResults.results(:,3).Performance.TrainingTime]';
    [val,idx] = min(allTTs(i));
    trainingTimes(i) = log(val);
    TT = NetLabels(i)+variants(idx);
    bestTT(i) = TT;
    % get the minimum MAE
    allMAEs(i,:) = [...
        networkResults.results(:,1).Performance.AveragePredictionError, ...
        networkResults.results(:,2).Performance.AveragePredictionError, ...
        networkResults.results(:,3).Performance.AveragePredictionError]';
    [val,idx] = min(allMAEs(i));
    MAE(i) = val;
    bMAE = NetLabels(i)+variants(idx);
    bestMAE(i) = bMAE;
    % get the maximum accuracy
    allAccs(i,:)=[...
        networkResults.results(:,1).Performance.Accuracy(1), ...
        networkResults.results(:,2).Performance.Accuracy(1), ...
        networkResults.results(:,3).Performance.Accuracy(1)]';
     [val,idx] = max(allAccs(i));
    accuracy(i) = val;
    bAcc = NetLabels(i)+variants(idx);
    bestAcc(i) = bAcc;
% Compute the RMSE for AlexNet models
    if i == 1
        for indexes = 1:3
            preds = networkResults.results(:,indexes).Predictions;
            tgts = networkResults.results(:,indexes).validationData;
            avpredErr = mean(abs(preds-tgts));
            ANet_RMSE(indexes) = sqrt(mean(abs(preds-tgts).^2));
        end
        AllRMSEs(1,:) = ANet_RMSE';
        val =  0;
        idx = 0;
        [val,idx] = min(ANet_RMSE);
        RooMeanSquareError(i) = val;
        bRMSE = NetLabels(i)+variants(idx);
        bestRMSE(i) = bRMSE;
    else
        val = 0;
        idx = 0;
        AllRMSEs(i,:) = [...
            networkResults.results(:,1).Performance.RootMeanSquareError,...
            networkResults.results(:,2).Performance.RootMeanSquareError,...
            networkResults.results(:,3).Performance.RootMeanSquareError,...
            ]';
        [val,idx] = min(AllRMSEs(i));
        bRMSE = NetLabels(i)+variants(idx);
        RootMeanSquareError(i) = val;
        bestRMSE(i) = bRMSE;
    end
    AllRMSEs(1,:) = ANet_RMSE';
    % save each model as a table row
    end
end
%plot best training times
figure(1); clf reset;
bar([1,2,3,4,5,6],trainingTimes);
ax1 = gca;
ax1.XTick = [1 2 3 4 5 6];
ax1.XTickLabels = bestTT;
ax1.XTickLabelRotation = 45;
ax1.YLim = [0 10];
title('Model Comparisons: Minimum Training Time');
ylabel('log(Training Time)');
% plot best MAE
figure(2); clf reset;
bar([1,2,3,4,5,6],MAE);
ax2 = gca;
ax2.YLim = [0 0.10];
ax2.XTickLabels = bestMAE;
ax2.XTickLabelRotation = 45;
title('Model Comparisons: Minimum Mean Absolute Error');
ylabel('Mean Absolute Error');
% plot best accuracy
figure(3); clf reset;
ax3 = gca;
bar([1,2,3,4,5,6],accuracy);
ax3.XTick = [1 2 3 4 5 6];
ax3.XTickLabels = bestMAE;
ax3.XTickLabelRotation = 45;
ax3.YLim = [0 1];
title('Model Comparisons: Maximum Accuracy_(_1_0_%_)');
ylabel('Average Accuracy (10% threshold))');
%
%manually insert the value of AlexNet's RMSE
RootMeanSquareError(1) = min(ANet_RMSE);
figure(4); clf reset;
ax4 = gca;
bar([1,2,3,4,5,6],RootMeanSquareError);
ax4.XTick = [1 2 3 4 5 6];
ax4.XTickLabels = bestRMSE;
ax4.XTickLabelRotation = 45;
ax4.YLim = [0 0.2];
title('Model Comparisons: Minimum Root Mean Square Error');
ylabel('Root Mean Square Error')

figure(5); clf reset;
ax5 = gca;
bar([1:6],log(allTTs));
title('Comparing CNN Model Training Times for All Training Variants');
ax5.XTick = [1 2 3 4 5 6];
ax5.XTickLabels = NetLabels;
ax5.XTickLabelRotation = 45;
ax5.YLim = [0 10];
legend(variants);
ylabel('log(Training Times)');
% 
figure(6); clf reset;
ax6 = gca;
bar([1:6],100*allAccs);
title('Comparing CNN Model Accuracies for All Training Variants');
ax6.XTick = [1 2 3 4 5 6];
ax6.XTickLabels = NetLabels;
ax6.XTickLabelRotation = 45;
ax6.YLim = [0 100];
legend(variants);
ylabel('Accuracy_(_1_0_%_)');
%
figure(7); clf reset;
ax7 = gca;
bar([1:6],AllRMSEs);
title('Comparing CNN Model RMSEs for All Training Variants');
ax7.XTick = [1 2 3 4 5 6];
ax7.XTickLabels = NetLabels;
ax7.XTickLabelRotation = 45;
ax7.YLim = [0 0.2];
legend(variants);
ylabel('Root Mean Square Error');
%
figure(8); clf reset;
ax8 = gca;
bar([1:6],allMAEs);
title('Comparing CNN Model MAEs for All Training Variants');
ax8.XTick = [1 2 3 4 5 6];
ax8.XTickLabels = NetLabels;
ax8.XTickLabelRotation = 45;
ax8.YLim = [0 0.12];
legend(variants);
ylabel('Mean Absolute Error');
