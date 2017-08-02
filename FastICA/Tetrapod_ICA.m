clc; clear all; close all;
%% Load linear combination from Tetrapod
load('C:\Users\orrav\Documents\Technion\8th\Project\Alex_Orr_Project\TetrapodPSF\Sequence2.mat');

%% Mixed Sources Matrix
size_lincomb = size(Sequence2.LinearCombinations) ;
y = zeros(size_lincomb(3),size_lincomb(1)*size_lincomb(2)) ;
for i=1:size_lincomb(3)
    y(i,:) = reshape(Sequence2.LinearCombinations(:,:,i), 1, []) ;
end

%% FastICA -- to obtain Mixing_mat and original sources
[A,W] = fastica(y) ;
s = W * y ;
s_img_mat = zeros(size_lincomb(1), size_lincomb(2), size_lincomb(3));
for j=1:size_lincomb(3)
    s_img_mat(:,:,j) = reshape(s(j,:), size_lincomb(1), size_lincomb(2));
    figure(j);
    imagesc(s_img_mat(:,:,j));
end
    