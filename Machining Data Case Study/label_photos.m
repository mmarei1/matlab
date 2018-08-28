% Develop a script that labels each image based on the following:
% experiment : exp01_,...exp25_
% cuttingLength: cL02,06,cl10,...,cl240;
% side: s01,s02
experiment = {};
imgFolder = fullfile('C:\Users\mareim\Documents\MATLAB\Machining Data Case Study\');
imgSet = imageSet(imgFolder);
fileDestination = 'ctd_experiment_files\';
filename = 'Cutting_tool_data_31625_image';
for i=1:25
   experiment{i} = sprintf('exp%02d',i) ;
end
% Assign index array labels
indexArray = [2,6,10,20,40,60,80,100,120,140,160,180,200,220,240];

clabels= {};
i = 1;
for i = 1:size(indexArray,2)
    clabels{i} = sprintf('_%03d',indexArray(i));
end

sides = {'_s1','_s2'};

% calculate the number of pics in each experiment
for i = 1:25
    img_in_exp(i) = 4 + 2*cv_ci(i,2)
end

totalImages = sum(img_in_exp)

% Asked for Ibrahim's help with this logic
% construct each experiment label
% for experiment e of 25
images = 1;

% create a label array with the label details
healthLabel = [];
imCounter = 1;
for e = 1:25
    side_counter = 1;
    % for img j in exp e
    for j = 1:img_in_exp(e)
        % if img_in_exp(e) - j <= 1
        if img_in_exp(e) - j <=1
            healthLabel(imCounter) = 0;
        else 
            healthLabel(imCounter) = 1;
        end
        imCounter = imCounter+1;
        
        fileAddr = [fileDestination,filename,sprintf('%03d',images),'.png'];
        if mod(images,2) ==1
            label = [experiment{e}, clabels{side_counter}, sides{1}, '.png'];
            disp(['label #' , num2str(images) +'created :', label])
            
           
        elseif mod(images,2) ==0
            label = [experiment{e},  clabels{side_counter}, sides{2}, '.png'];
            disp(['label #' , num2str(images) +'created :', label])   
        end % end of ifel
        % create the new file label
        newFileLabel = [fileDestination label];
        disp(['File to be moved: ',fileAddr])
        disp(['Destination: ', newFileLabel])
        disp(';')
        % delete above code and reuse to assign new label of 'healthy' or
        % 'worn' based on previous image 
        images = images + 1;
        if mod(j,2) ==0
            side_counter = side_counter+1;
        end % end of if 2
    end % end of for loop 2
end % end of for loop 1


% label the health state based on the position of the image in relation to
% the number of images in each experiment - the last two images of each
% experiment are of the failed tool - everything else is "healthy"

for e = 1:25
    side_counter = 1;
    % for img j in exp e
    for j = 1:img_in_exp(e)
        % if img_in_exp(e) - j <= 1
        if img_in_exp(e) - j <=1
            healthLabel(imCounter) = 0;
        else 
            healthLabel(imCounter) = 1;
        end
        imCounter = imCounter+1;
    end
end

% end of script