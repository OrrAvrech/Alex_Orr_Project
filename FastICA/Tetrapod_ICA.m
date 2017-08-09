clc; clear all; close all;
%% Load linear combination from Tetrapod
load('..\TetrapodPSF\Sequence2.mat');

%% Mixed Sources Matrix
size_lincomb = size(Sequence2.LinearCombinations) ;
x = zeros(size_lincomb(3), size_lincomb(1)*size_lincomb(2)) ;
for i=1:size_lincomb(3)
    x(i,:) = reshape(Sequence2.LinearCombinations(:,:,i), 1, []) ;
end
%% Movie
for i=1:size_lincomb(3)
    imagesc(Sequence2.LinearCombinations(:,:,i));
    pause (1) ;
end
%% FastICA -- to obtain Mixing_mat and original sources
num_planes = 2;
[A,W] = fastica(x, 'numOfIC', num_planes) ;
s = W * x;
s_img_mat = zeros(size_lincomb(1), size_lincomb(2), num_planes);
for j=1:num_planes
    s_img_mat(:,:,j) = reshape(s(j,:), size_lincomb(1), size_lincomb(2));
    figure(j);
    imagesc(s_img_mat(:,:,j));
end
    