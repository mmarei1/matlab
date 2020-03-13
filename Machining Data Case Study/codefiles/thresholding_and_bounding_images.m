% script to segment image
% created by Mohamed Marei on 15/10/18
filepath = 'C:\Users\mareim\Documents\MATLAB\NewSandbox\Machining Data Case Study\NewImages\Edge\Healthy\';
basename = 'A24-W-20M[100]-1.jpg';
imageName = [filepath 'A24-W-80M[100]-1.jpg'];
maskName = [filepath 'A25-W-20M[100]-1.jpg'];
I = imread(imageName);
% ISim = imread(maskName);
% create a matrix of 2x2 subplots to display the image
figure(1); clf reset;
subplot(2,4,1);
imshow(I), title('Original Image');
%subplot(2,4,8);
%imshow(rgb2gray(I) & rgb2gray(ISim));
% % obtain grayscale equivalent of the image
% IG = rgb2gray(I);
% % Step 2: implement edge detection and thresholding
% [~, threshold] = edge(IG, 'sobel');
% fudgeFactor = .7;
% BWs = edge(IG,'sobel', threshold * fudgeFactor);
% figure, imshow(BWs), title('binary gradient mask');

% convert image to HSV equivalent
hsvImage = rgb2hsv(I);
hImage = hsvImage(:, :, 1);
sImage = hsvImage(:, :, 2);
vImage = hsvImage(:, :, 3);
subplot(2,4,5);
imhist(hImage), title('Hue Histogram');
subplot(2,4,6);
imhist(sImage) , title('Saturation Histogram');
subplot(2,4,7);
imhist(vImage) , title('Value Histogram');

diffImage = (sImage - hImage);
subplot(2,4,4);
imshow(diffImage);
% most images have been placed against a black padded bg, making the black
% very distinct from the surrounding image patch (if selected based on value)
foreground = vImage >= 0.1;
% extract largest blob
% foreground = bwareafilt(foreground,2);
foreground = bwpropfilt(foreground,'Perimeter',1);

% extract the foreground elements 
labeledImage = bwlabel(foreground);
subplot(2,4,2);
imshow(foreground), title('Image Mask (Saturation)');

measurements = regionprops(labeledImage,'BoundingBox');
bb = measurements.BoundingBox;
% add a padding of 20 px to each measurement
% note the cropping bounding box is specified as [X_min Y_min W H]
bb
% As we are only interested in one region of the image, we can cleverly
% shift the image vertically upwards and to the left if the bounding box
% dimensions don't correspond to the right focus region. Otherwise, do not
% shift or pad the image
vShift = 120;
hShift = 120;
vPad = -100;
hPad = -100;
if bb(2)<= 300
    vShift = 120;
    hShift = 0;
    vPad = -160;
    hPad = 0;
end
bbA = bb + [hShift vShift hPad vPad]
roiImage = imcrop(I, bbA);
%filtered_Image = imfilter(roiImage,foreground);

subplot(2,4,3);
imshow(roiImage), title('Cropped Image');;

% subplot(2,4,8);
% horizontalProfile = sum(imbinarize(I),1);
% verticalProfile = sum(imbinarize(I),2);
% topRow = find(verticalProfile, 1, 'first');
% bottomRow = find(verticalProfile, 1, 'last');
% leftColumn = find(horizontalProfile, 1, 'first');
% rightColumn = find(horizontalProfile, 1, 'last');
% binaryImage = rgb2gray(I) > 100;
% croppedImage = binaryImage(topRow:bottomRow, leftColumn:rightColumn);
% imshow(croppedImage);
bw3 = edge(rgb2gray(I),'sobel',0.016);
subplot(2,4,8);

foreground2 = bwareafilt(bw3,1);
subplot(2,4,4);
imshow(foreground2);

% obtain new bounding box
labeledImage = bwlabel(foreground2);
measurements = regionprops(labeledImage,'BoundingBox');
bbN = measurements.BoundingBox + [-20 -20 20 20];
recropped = imcrop(I,bbN);
subplot(2,4,8);
imshow(recropped);

% plot the final cropped image on a new set of axes
figure(3); clf reset;
imshow(recropped)

figure(4); clf reset;
imshow(roiImage), title('Final cropped image')