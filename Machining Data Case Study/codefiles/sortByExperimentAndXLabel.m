% script to output a new table sorted by experiment number and the length
% of wear
load cuttingParametersTable.mat
load localTable.mat
startIndex = 1;
endIndex = 0;
speeds = reshape(repmat(cuttingSpeeds,5,1),25,1);
sortedLocalTable = table('Size',[553,9],'VariableTypes',{'double','double','string','double','double','double','double','double','double'},'VariableNames',{'id','exp','filename','vc','fd','ae','rrc','xlabel','ylabel'});
for i = 1:numel(unique(localTable.exp))
    % set the current experiment to i
    currentExp = i;
    % take all the sub-experiments under i to be the new experiments
    expRows = localTable.exp == currentExp;
    subTable = localTable(expRows,:);
    numRows = height(subTable);
    %subTable = localTable(expRows,:);
    %oldIDs = localTable.id(expRows);
    newSubtable = sortrows(subTable,'xlabel');
    sprintf('exp: %d,  numRows: %d', currentExp,numRows)
    endIndex = startIndex+numRows-1;
    newIDs = startIndex:endIndex;
    ae = cuttingParametersTable.CuttingWidth(i)*ones(numel(newIDs),1)
    fd = cuttingParametersTable.feedRate(i)*ones(numel(newIDs),1)
    rrc = cuttingParametersTable.rrcTool(i)*ones(numel(newIDs),1)
    vc = speeds(currentExp)*ones(numel(newIDs),1);
% assign parameters individually
    sidx = newIDs(1)-startIndex+1;
    sortedLocalTable.id(newIDs) = newIDs;
    sortedLocalTable.exp(newIDs) = currentExp*ones(numRows,1);
    sortedLocalTable.filename(newIDs) = newSubtable.filename(:);
    sortedLocalTable.vc(newIDs) = vc;
    sortedLocalTable.fd(newIDs) = fd;
    sortedLocalTable.ae(newIDs) = ae;
    sortedLocalTable.rrc(newIDs) = rrc;
    sortedLocalTable.xlabel(newIDs) = newSubtable.xlabel(:);
    sortedLocalTable.ylabel(newIDs) = newSubtable.ylabel(:);
    %,newSubtable.exp(newIDs),newSubtable.filename(newIDs),fd(newIDs),ae(newIDs),rrc(newIDs),newSubtable.xlabel(newIDs),newSubtable.ylabel(newIDs)];
    
    
    startIndex = endIndex+1;
end

% save the resulting table to use in our prediction framework
save('sortedLocalTable.mat','sortedLocalTable');

%localTable = sortedLocalTable;