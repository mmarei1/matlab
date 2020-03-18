function handles = prepareLossPlots(numExperiments)
%PREPAREPLOTS creates a figure to display training progress information.
    colors = lines(numExperiments);
    f = figure('Units','normalized','Position',[0.1 0.1 0.5 0.5]);
    f.Visible = true;
    subplot(2,1,1), ylabel('Accuracy'), grid on;
    subplot(2,1,2), ylabel('Loss'), xlabel('Iteration'), grid on;
    for i=1:numExperiments
        subplot(2,1,1);
        handles.trainingRMSELines(i) = animatedline('Color',colors(i,:),'LineStyle','-');
        handles.validationRMSELines(i) = animatedline('Color',colors(i,:),'LineStyle','none','Marker','.','MarkerSize',14);
        subplot(2,1,2)
        handles.trainingLossLines(i) = animatedline('Color',colors(i,:),'LineStyle','-');
        handles.validationLossLines(i) = animatedline('Color',colors(i,:),'LineStyle','none','Marker','.','MarkerSize',14);
    end
    subplot(2,1,1), legend(handles.rmseLines,string(1:numExperiments));
    subplot(2,1,2), legend(handles.lossLines,string(1:numExperiments));
end