% function to caluclate class penalties from data labels.
% out = calculateClassPenalties(data) 
% returns a vector of classification weights calculated from the frequency
% of each input.
% Created by Mohamed Marei, 2020.
%__________________________________________________________________________
function pens = calculateClassPenalties(data)
dataLabels = tabulate(data);
classFreqs = cell2mat(dataLabels(:,3));
classPenRatios = 1./classFreqs;
pens = ceil(classPenRatios./min(classPenRatios));
end
