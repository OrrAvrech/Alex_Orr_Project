function [ error ] = ImgXcorr( ImgOrig, ImgIC )

% Calculate Euclidian distance between images and normalize
% N = numel(model)*sqrt(200^2+200^2);
% Dist = sum(sqrt((model(:) - data(:))).^2);
% error = Dist/N;

% Calculate cross correlation between images
corr  = xcorr2(ImgOrig, ImgIC);
% mean_corr = mean(mean(corr));
max_corr  = max(corr);
error = 1 - (mean(max_corr))/(sum(sum(corr)));

end

