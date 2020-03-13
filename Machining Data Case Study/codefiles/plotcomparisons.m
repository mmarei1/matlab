    figure(2); clf reset;
    plot(1:numel(yval),yval,'b+',"LineWidth",2);
    hold on;
    kp = [0.1:0.10:1];
    ypred_kernelised = cell(numel(kp));
    for i = 1:numel(kp)
        ypred_kernelised{i} = abs(YPred{1}).^kp(i);
        [mae1,rmse1,a10,a20,a30] = benchmarkModelSkill(ypred_kernelised{i},yval);
        scatter(1:numel(yval),ypred_kernelised{i});
        legendLabels{i} = sprintf("Predictions (kp = %2f)",kp(i));
    end
    %plot(1:numel(yval),ypred_re2_n,'ro');
     
    hold off;
    legend(["Targets",legendLabels])
    ylabel("Normalised Tool Wear Value (mm)")
    xlabel("Record Number")
    %title(sprintf("Finetuning vs Feature Extraction methods for %s - RMSE_ft = %3f; RMSE_fe = %3f",expNames{i},rmse_ens2_n, rmse1))
    predictions_plotname = expNames{1}+"_comparison_"+".png";
    %predictions_filename = expNames{i}+"_predictions.mat";
    saveas(gcf,predictions_plotname);