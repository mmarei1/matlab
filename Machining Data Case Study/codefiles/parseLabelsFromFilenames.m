function [expLabel, cutDistLabel, magLabel,imVariantLabel]=parseLabelsFromFilenames(filename,imageDir)
    % extract between label parts, specified as delimiters
    % expLabel = experiment label (1,2,3,...,25)
    % cutDistLabel = 0.2M, 0.6M, 1M, ....., 240M
    % magLabel = 50x, 100x, ..., 200x
    % imVariant = 1,2,3,....
    % '/home/mareim/Downloads/cutting_tool_images/01/A1-W-0.2M[100]-1.jpg'
    fn = strcat(filename,"");
    expLabel = str2num(extractBetween(fn,"/A","-W-","Boundaries","exclusive"));
    cutDistLabel = str2num(extractBetween(fn,"-W-","M[","Boundaries","exclusive"));
    magLabel = str2num(extractBetween(fn,"[","]","Boundaries","inclusive"));
    imVariantLabel = str2num(extractBetween(fn,"]-",".jpg","Boundaries","exclusive"));
end