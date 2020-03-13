% from the image datastore file names, we need the substrings that are
% unique to each file
clear newNames cuttingLengthLabels;
imageNames = imds.Files;
trimStart= '\A';
trimEnd = '.jpg';
experiment_record = 1;

newDataTable = table([1:553]',zeros(553,1),string(sort(imageNames)),zeros(size(newNames,1),1),zeros(size(newNames,1),1),'VariableNames',{'id','exp','filename','xlabel','ylabel'})

% Assigning the values in 
for i = 1:size(imageNames,1)
    trimmedImageName = extractBetween(imageNames{i},trimStart,trimEnd);
    % check if the image has a corresponding wear value
    %check if each extracted string name has a leading zero - if not, add
    %it
    expNum = str2double(extractBetween(string(trimmedImageName),"","-W"));
    % preformat prefix for %2f; 
    % if preformatted string == experimentNumber,
    % don't replace
    old_prefix = string(expNum+"-");
    new_prefix = sprintf("A%0.2d-",expNum);
    % code does not behave as expected because it replaces all matching
    % instances
    modifiedImageName = strrep(string(trimmedImageName),old_prefix,new_prefix);
    % pad cutting length label as well to ensure correct sorting results
    cl_label_value = str2double( extractBetween(modifiedImageName,"W-","M"));
    
    % search the distances cell array for the current experiment number
    currentExp = distances{expNum};
    
    % logical array matching the cutting length value in the current experiment
    cl_tf = (currentExp == cl_label_value);
    
    % doesn't work as expected because this is never empty
    if isempty(cl_tf)
        cl_val = 0;
        cuttingLengthLabels(i) = cl_val;
    else
        cl_val = currentExp(cl_tf);
        cuttingLengthLabels(i) = cl_val;
    end
    
    
    % if the cl_label_value matches a value in currentExp
    
    newNames{i} = modifiedImageName;
    
end
newNames =  (string(newNames))';

% % read each image string to extract experiment number and cutting length
% for i = 1:size(trimmedImageNames,1)
%     % experiment record to search
%     experiment_to_match = "A" + experiment_record + "-";
%     % select the subset of image names that matches the current experiment
%     hits = strfind(string(trimmedImageNames),experiment_to_match);
%     % wear values between 0<=l<=10 are excluded
%     hitIndexes = find(not(cellfun('isempty',hits)));
%     experiment = trimmedImageNames(hitIndexes);
%     for ex = 1:size(experiment,1)
%         cuttingLengthLabels(ex) = str2double(cell2mat(extractBetween(tin{1},"A1-W-","M[100]-1.jpg")));
%         % check if this cutting length corresponds to a value in our cutting lengths table
%         if cuttingLengthLabels(ex) == cell2mat(distances{i})
%             indexInCuttingDistance = find(distances,cuttingLengthLabels)
%         else
%             
%         end
%     end
% end