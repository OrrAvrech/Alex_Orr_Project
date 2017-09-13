function [ error ] = ImgXcorr( ImgOrig, ImgIC )

corr  = xcorr2(ImgOrig, ImgIC);
% mean_corr = mean(mean(corr));
max_corr  = max(corr);
error = 1 - (mean(max_corr))/(sum(sum(corr)));

end

