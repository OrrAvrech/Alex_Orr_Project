function [ error ] = ImgL2( ImgOrig, ImgIC )
% calculate Euclidian distance between images and normalize
N = numel(ImgOrig)*sqrt(200^2+200^2);
Orig = reshape(ImgOrig, 1,[]);
IC = reshape(ImgIC, 1,[]);
Dist = sum((Orig - IC)).^2;
error = Dist/N;

end

