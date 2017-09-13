function [ error ] = euc_dist( ImgOrig, ImgIC )
% calculate Euclidian distance between mass centers and normalize
N = sqrt(200^2+200^2);
[X_ica,Y_ica] = mass_cent(ImgIC);
[x,y] = mass_cent(ImgOrig);
it_dist = sqrt((X_ica - x).^2+(Y_ica - y).^2);
error = it_dist/N;

end

