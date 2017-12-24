function [ error ] = ImgXcorr( ImgOrig, ImgIC )

% Calculate cross correlation between images
corr  = xcorr2(ImgOrig, ImgIC)/(std(ImgOrig(:))*std(ImgIC(:)));
max_corr  = max(corr(:));
error = max_corr;

end

