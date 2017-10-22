function [ error ] = ImgL2( model, data )
% calculate Euclidian distance between images and normalize
% N = numel(model)*sqrt(200^2+200^2);
% Dist = sum(sqrt((model(:) - data(:))).^2);
% error = Dist/N;

% error = sum((model(:)-data(:)).^2)/(200*200); % MSE

N = numel(model);
error = sum(model(:) - data(:).*log(model(:))) ./ N;

end

