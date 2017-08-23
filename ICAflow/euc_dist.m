function [ error ] = euc_dist( X_orig,Y_orig,Im )
% calculate Euclidian distance between mass centers and normalize
N = sqrt(200^2+200^2);
[X_ica,Y_ica] = mass_cent(Im);
[x,y] = pix_metric(X_orig,Y_orig,0);
it_dist = sqrt((X_ica - x).^2+(Y_ica - y).^2);
error = it_dist/N;

end

