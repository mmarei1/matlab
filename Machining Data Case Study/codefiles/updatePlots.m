function updatePlots(handles,data)
%UPDATEPLOTS updates the plots in handles, created by PREPAREPLOTS, with
%training progress information.
    info = data.info;
    if info.State == "iteration"
        addpoints(handles.accuracyLines(data.experimentNumber),info.Iteration,info.TrainingAccuracy);
        if ~isempty(info.ValidationAccuracy)
            addpoints(handles.validationAccuracyLines(data.experimentNumber),info.Iteration,info.ValidationAccuracy);
        end
        addpoints(handles.lossLines(data.experimentNumber),info.Iteration,info.TrainingLoss);
        if ~isempty(info.ValidationLoss)
            addpoints(handles.validationLossLines(data.experimentNumber),info.Iteration,info.ValidationLoss);
        end
    end
end
