% Preprocessing images to improve performance for VGG16

% Step 1- subtract the mean value of each image
% Step 2- smoothing with a Gaussian kernel stdev 3px
% Step 3- rectified image reference frame
% Step 4- resizing images to 224 by 224 pixels

function preprocessedImage = preprocessImage(im)
    
    % Step 1: subtract the mean value of each image
    % isolate rgb channels
    rc = im(:,:,1);
    bc = im(:,:,2);
    gc = im(:,:,3);
    % calculate the mean of each channel
    rm = mean2(rc);
    bm = mean2(bc);
    gm = mean2(gc);
    % concatenate mean subtracted channels along dimension 3 
    meanAdjustedImage = cat(3, rc - rm, bc - bm, gc - gm);
    % output the new image with a stdev of 3
    filteredImage = imgaussfilt(meanAdjustedImage,2);
    
    % Step 2: image smoothing with a Gaussian Kernel
    % define Gaussian kernel smoothing function
    preprocessedImage = filteredImage;
end

% additional functions to implement Gaussian kernel smoothing
function fhwm = sigma2fhwm(sigma)
    fhwm = sigma *sqrt(3*log10(2));
end

function sigma = fhwm2sigma(fhwm)
    sigma = fhwm / sqrt(3*log10(2));
end