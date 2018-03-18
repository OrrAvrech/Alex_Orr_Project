function [ error ] = ImgXcorr( ImgOrig, ImgIC )

% Calculate cross correlation between images
corr  = xcorr2(ImgOrig, ImgIC); % Normalized CC
max_corr  = max(corr(:)); % TODO
error = max_corr;

end

