clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures if you have the Image Processing Toolbox.
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 18;

%===============================================================================
% Read in a color demo image.
folder = 'C:\Users\mareim\Documents\MATLAB\NewSandbox\Machining Data Case Study\NewImages\Edge\Healthy\';
baseFileName = 'A25-W-40M[100]-1.jpg';
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
rgbImage = imread(fullFileName);
% Get the dimensions of the image.  numberOfColorBands should be = 3.
[rows, columns, numberOfColorBands] = size(rgbImage);
% Display the original color image.
subplot(3, 4, 1);
imshow(rgbImage);
axis on;
title('Original Color Image', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

hsv = rgb2hsv(rgbImage);
h = hsv(:,:,1);
s = hsv(:,:,2);
v = hsv(:,:,3);

% Display the individual color channel images.
subplot(3, 3, 2);
imshow(h, []);
title('Hue Channel Image', 'FontSize', fontSize);
subplot(3, 3, 3);
imshow(s, []);
title('Saturation Channel Image', 'FontSize', fontSize);
subplot(3, 3, 4);
imshow(v, []);
title('Value Channel Image', 'FontSize', fontSize);

% Let's compute and display the histograms.
[pixelCount, grayLevels] = imhist(h, 500);
subplot(3, 3, 6); 
bar(grayLevels, pixelCount);
grid on;
title('Histogram of Hue Image', 'FontSize', fontSize);
xlim([0 0.2]); % Scale x axis manually.
[pixelCount, grayLevels] = imhist(s);
subplot(3, 4, 7); 
bar(grayLevels, pixelCount);
grid on;
title('Histogram of Saturation Image', 'FontSize', fontSize);
xlim([0 grayLevels(end)]); % Scale x axis manually.
[pixelCount, grayLevels] = imhist(v);
subplot(3, 3, 8); 
bar(grayLevels, pixelCount);
grid on;
title('Histogram of Value Image', 'FontSize', fontSize);
xlim([0 grayLevels(end)]); % Scale x axis manually.

% Threshold based on Hue Channel
% thresholdValue = graythresh(h) % Otsu method is not so good.  Better to use fixed threshold.
thresholdValue = 0.075;
binaryImageH = h < thresholdValue;
% Show as line on bar chart
subplot(3, 3, 6);
yl = ylim;
line([thresholdValue, thresholdValue], [yl(1), yl(2)], 'Color', 'r', 'LineWidth', 2);
% Fill holes to get rid of reflection.
binaryImageH = imfill(binaryImageH, 'holes');
% Get rid of blobs less than 100 pixels in size.
binaryImageH = bwareaopen(binaryImageH, 100);
subplot(3, 3, 9);
imshow(binaryImageH, []);
axis on;
title('Binarized Hue Channel Image', 'FontSize', fontSize);
% Threshold based on Value Channel
% binaryImageV = v < .5;
% subplot(3, 4, 12);
% imshow(binaryImageV, []);
% axis on;
% title('Binarized Value Channel Image', 'FontSize', fontSize);

area1 = bwarea(binaryImageH)
area2 = sum(binaryImageH(:))
message = sprintf('Area of binary Hue image = \n%.2f, or %d pixels\nDepending on method used.', area1, area2);
fprintf('%s\n', message);
uiwait(helpdlg(message));