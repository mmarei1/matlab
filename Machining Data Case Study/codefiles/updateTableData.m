function ut = updateTableData(table)
    % modify each filename in the table
    altfilepath = 'C:\Users\Mareim\OneDrive - Coventry University\Data\NewImages\EdgeNew';
    altfilesuffix = '_newImage.jpg';
    
    updatedFileName = table.filename;
end

ut = strrep(updatedDataTable.filename(:),'Edge','EdgeNew');

% userpath
user_path = "C:\Users\ThalesHPC\OneDrive - Coventry University\NewImages";

% replace specified directory in filename column with local user directory
%resizedImagesTable_remote = resizedImagesTable;
