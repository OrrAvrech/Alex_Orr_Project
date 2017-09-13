function [ error ] = ImgL2( ImgOrig, ImgIC )
% calculate Euclidian distance between images and normalize
N = numel(ImgOrig)*sqrt(200^2+200^2);
Dist = norm(ImgOrig - ImgIC);
error = Dist/N;

end

