function [ error ] = PoissCost( model, data )

% Calculate error using the Poisson Cost function
% Normalization should be optimized

N = numel(model);
error = sum(model(:) - data(:).*log(model(:))) ./ N;

end

