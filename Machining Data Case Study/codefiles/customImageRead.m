function outImage = customImageRead(imPath,tgtSize)
    
    [Img, cmap] = imread(imPath);
    Img_RGB = ind2rgb(Img,cmap);
%     i_mean(1) = mean(i1);
%     i_mean(2) = mean(i2);
%     i_mean(3) = mean(i3);
%     i1_std = stdev(i1);
%     i2_std = stdev(i2);
%     i3_std = stdev(i3);
    Img_c = zeros(size(Img_RGB),'like',Img_RGB);
    for i = 1:3
        i_mean(i) = mean(Img_RGB(:,:,i),'all');
        i_stdv(i) = std(Img_RGB(:,:,i),0,'all');
        Img_c(:,:,i) = (Img_RGB(:,:,i) - i_mean(i) )./i_stdv(i);
    end
    
    wcrop = centerCropWindow2D(size(Img_c,[1,2]),[600,600]);
    Img_cropped = imcrop(Img_c,wcrop);
    outImage = imresize(Img_cropped, tgtSize);  
end