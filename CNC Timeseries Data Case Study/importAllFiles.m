function [XTrain, YTrain, XTest, YTest] = importAllFiles(filenames)
    % import all files in filenames
    
    for i = 1:size(filenames,1)
        % output two tables, 1 with pre-normalised data and 1 with
        % normalised data
        [tsdata, ndata] = importfile1(filenames(i));
        tsdata.Index = i;
        trimmedname = erase(filename(i),".csv");
        tsdata.Datetime = datetime(trimmedname);
        % place at the end of a new table
        % if not the first row index
        len = size(ndata,1);
        if i == 1
            firstIndex = 1;
            lastIndex = len;
        else 
            firstIndex = 1 + lastIndex;
            lastIndex = firstIndex + len;
        end
        Xdata(firstIndex:lastIndex) = ndata;
        Ydata(firstIndex:lastIndex) = [tsdata(:,1),normalise(tsdata(:,4),[0,1])];
    end
    
end
