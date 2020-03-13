results_alexnet = load('results-AlexNet.mat','results');
results_resnet18 = load('results-ResNet-18.mat','results');
results_resnet50 = load('results-ResNet-50.mat','results');
results_resnet101 = load('results-ResNet-101.mat','results');
results_inceptionv3 = load('results-InceptionV3.mat','results');
results_squeezenet = load('results-SqueezeNet.mat','results');
%%
% take the validation data and plug it into a vector
valdata = results_alexnet.results(1).validationData;

legendLabels = ["targets","AlexNet","InceptionV3","ResNet-18","ResNet-50","ResNet-101","SqueezeNet"];
optLabels = ["ADAM","RMSPROP","SGDM"]
predictions_adam = {
    valdata;
    results_alexnet.results(1).Predictions;
    results_inceptionv3.results(1).Predictions;
    results_resnet18.results(1).Predictions;
    results_resnet50.results(1).Predictions;
    results_resnet101.results(1).Predictions;
    results_squeezenet.results(1).Predictions;
    };

predictions_sgdm = {
    valdata;
    results_alexnet.results(3).Predictions;
    results_inceptionv3.results(3).Predictions;
    results_resnet18.results(3).Predictions;
    results_resnet50.results(3).Predictions;
    results_resnet101.results(3).Predictions;
    results_squeezenet.results(3).Predictions;};

predictions_rmsprop = {
    valdata;
    results_alexnet.results(2).Predictions;
    results_inceptionv3.results(2).Predictions;
    results_resnet18.results(2).Predictions;
    results_resnet50.results(2).Predictions;
    results_resnet101.results(2).Predictions;
    results_squeezenet.results(2).Predictions;
    };
%%
optLabels = ["Target Data","Predictions (ADAM)","Predictions (RMSPROP)","Predictions  (SGDM)"];


for i = 1:6
%subplot(2,3,i)
figure(i); clf reset;
plot([1:numel(valdata)],predictions_adam{1},'bo','LineWidth',2,'MarkerSize',14);
hold on;
%for i=1:3
plot([1:numel(valdata)],predictions_adam{i+1},'+','LineWidth',2,'MarkerSize',14);
plot([1:numel(valdata)],predictions_rmsprop{i+1},'x','LineWidth',2,'MarkerSize',14);
plot([1:numel(valdata)],predictions_sgdm{i+1},'d','LineWidth',2,'MarkerSize',14);
legend(optLabels)
set(gca,'FontSize',20)
hold off
xlabel("Observation Number",'FontSize',20)
ylabel("Normalized flank wear width (mm)",'FontSize',20)
title(sprintf("Prediction Outputs for CNN: %s",legendLabels(i+1)),'FontSize',20)
colormap('gray')
end

