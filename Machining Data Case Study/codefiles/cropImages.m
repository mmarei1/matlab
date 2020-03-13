% Demo to divide an image up into blocks (non-overlapping tiles).
% The first way to divide an image up into blocks is by using mat2cell().
% In this demo, I demonstrate that with a color image.
% Another way to split the image up into blocks is to use indexing.
% In this demo, I demonstrate that method with a grayscale image.
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
workspace;  % Make sure the workspace panel is showing.
fontSize = 20;
% % Read in a standard MATLAB color demo image.
% folder = fullfile(matlabroot, '\toolbox\images\imdemos');
% baseFileName = 'peppers.png';
% % Get the full filename, with path prepended.
% fullFileName = fullfile(folder, baseFileName);
% if ~exist(fullFileName, 'file')
% 	% Didn't find it there.  Check the search path for it.
% 	fullFileName = baseFileName; % No path this time.
% 	if ~exist(fullFileName, 'file')
% 		% Still didn't find it.  Alert user.
% 		errorMessage = sprintf('Error: %s does not exist.', fullFileName);
% 		uiwait(warndlg(errorMessage));
% 		return;
% 	end
% end
imfile = 'A7-W-80M[100]-1.jpg';

% Read the image from disk.
rgbImage = imread(imfile);
% Test code if you want to try it with a gray scale image.
% Uncomment line below if you want to see how it works with a gray scale image.
% rgbImage = rgb2gray(rgbImage);
% Display image full screen.
imshow(rgbImage);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
drawnow;
% Get the dimensions of the image.  numberOfColorBands should be = 3.
[rows columns numberOfColorBands] = size(rgbImage)
%==========================================================================
% The first way to divide an image up into blocks is by using mat2cell().
blockSizeR = 4; % Rows in block.
blockSizeC = 3; % Columns in block.
% Figure out the size of each block in rows. 
% Most will be blockSizeR but there may be a remainder amount of less than that.
wholeBlockRows = floor(rows / blockSizeR);
blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];
% Figure out the size of each block in columns. 
wholeBlockCols = floor(columns / blockSizeC);
blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];
% Create the cell array, ca.  
% Each cell (except for the remainder cells at the end of the image)
% in the array contains a blockSizeR by blockSizeC by 3 color array.
% This line is where the image is actually divided up into blocks.
if numberOfColorBands > 1
	% It's a color image.
	ca = mat2cell(rgbImage, blockVectorR, blockVectorC, numberOfColorBands);
else
	ca = mat2cell(rgbImage, blockVectorR, blockVectorC);
end
% Now display all the blocks.
plotIndex = 1;
numPlotsR = size(ca, 1);
numPlotsC = size(ca, 2);
for r = 1 : numPlotsR
	for c = 1 : numPlotsC
		fprintf('plotindex = %d,   c=%d, r=%d\n', plotIndex, c, r);
		% Specify the location for display of the image.
		subplot(numPlotsR, numPlotsC, plotIndex);
		% Extract the numerical array out of the cell
		% just for tutorial purposes.
		rgbBlock = ca{r,c};
		imshow(rgbBlock); % Could call imshow(ca{r,c}) if you wanted to.
		[rowsB columnsB numberOfColorBandsB] = size(rgbBlock);
		% Make the caption the block number.
		caption = sprintf('Block #%d of %d\n%d rows by %d columns', ...
			plotIndex, numPlotsR*numPlotsC, rowsB, columnsB);
		title(caption);
		drawnow;
		% Increment the subplot to the next location.
		plotIndex = plotIndex + 1;
	end
end
% Display the original image in the upper left.
subplot(4, 6, 1);
imshow(rgbImage);
title('Original Image');
%==============================================================================
% Another way to split the image up into blocks is to use indexing.
% Read in a standard MATLAB gray scale demo image.
folder = fullfile(matlabroot, '\toolbox\images\imdemos');
baseFileName = 'cameraman.tif';
fullFileName = fullfile(folder, baseFileName);
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
if ~exist(fullFileName, 'file')
	% Didn't find it there.  Check the search path for it.
	fullFileName = baseFileName; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
grayImage = imread(fullFileName);
% Get the dimensions of the image.  numberOfColorBands should be = 1.
[rows columns numberOfColorBands] = size(grayImage);
% Display the original gray scale image.
figure;
subplot(2, 2, 1);
imshow(grayImage, []);
title('Original Grayscale Image', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
% Divide the image up into 4 blocks.
% Let's assume we know the block size and that all blocks will be the same size.
blockSizeR = 128; % Rows in block.
blockSizeC = 128; % Columns in block.
% Figure out the size of each block. 
wholeBlockRows = floor(rows / blockSizeR);
wholeBlockCols = floor(columns / blockSizeC);
% Preallocate a 3D image
image3d = zeros(wholeBlockRows, wholeBlockCols, 3);
% Now scan though, getting each block and putting it as a slice of a 3D array.
sliceNumber = 1;
for row = 1 : blockSizeR : rows
	for col = 1 : blockSizeC : columns
		% Let's be a little explicit here in our variables
		% to make it easier to see what's going on.
		row1 = row;
		row2 = row1 + blockSizeR - 1;
		col1 = col;
		col2 = col1 + blockSizeC - 1;
		% Extract out the block into a single subimage.
		oneBlock = grayImage(row1:row2, col1:col2);
		% Specify the location for display of the image.
		subplot(2, 2, sliceNumber);
		imshow(oneBlock);
		% Make the caption the block number.
		caption = sprintf('Block #%d of 4', sliceNumber);
		title(caption, 'FontSize', fontSize);
		drawnow;
		% Assign this slice to the image we just extracted.
		image3D(:, :, sliceNumber) = oneBlock;
		sliceNumber = sliceNumber + 1;
	end
end
% Now image3D is a 3D image where each slice, 
% or plane, is one quadrant of the original 2D image.
msgbox('Done with demo!  Check out the two figures.');