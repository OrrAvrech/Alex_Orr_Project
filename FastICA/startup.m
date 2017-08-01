clc; clear all; close all;
%% Signal Separation
% Creating mixed signals
t = 0:0.001:20 ;
f1 = sin(t) ;
f2 = 0.5*t - floor(0.5*t) ;
figure(1) ;
subplot 211
plot(t,f1) ;
subplot 212
plot(t,f2) ;

s = [f1; f2] ;
A_real = [2 3 ; 1 5] ;
x = A_real*s ;

figure(2) ;
subplot 211
plot(t, x(1,:)) ;
subplot 212
plot(t, x(2,:)) ;
% ICA
[A,W] = fastica(x) ;
s_est = W*x ;
figure(3) ;
subplot 211
plot(t, s_est(1,:)) ;
subplot 212
plot(t, s_est(2,:)) ;

%% Image Separation
% Creating mixed images (nothing political)
fig1 = imread('PresidentObamaGS256x256.png') ;
fig2 = imread('mandrill_gray.png') ;
figure(1) ;
subplot 211
imshow(fig1) ;
subplot 212
imshow(fig2) ;

fig1_resh = reshape (fig1, 1, []) ;
fig2_resh = reshape (fig2, 1, []) ;
s = [fig1_resh; fig2_resh] ;
A_real = [0.2 0.3 ; 0.1 0.5] ;
x = A_real*double(s) ;
x1 = reshape(uint8(x(1,:)), 256, 256) ;
x2 = reshape(uint8(x(2,:)), 256, 256) ;

figure(2) ;
subplot 211
imshow(x1) ;
subplot 212
imshow(x2) ;
% ICA
[A,W] = fastica(x) ;
s_est = W * double(x) ;
s_est1 = reshape(s_est(1,:), 256, 256) ;
s_est2 = reshape(s_est(2,:), 256, 256) ;
figure(3) ;
subplot 211
imshow(s_est1) ;
subplot 212
imshow(s_est2) ;







