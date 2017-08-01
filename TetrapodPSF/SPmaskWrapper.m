clear;
close all;
%% Load masks
load('SPmask4um.mat','maskBest');
%% Set parameters
nomFocusVec=0; % focal plane (0 = interface)
n=10;
zVec = linspace(-2e-6,2e-6,n+1); % z positions of emitter
pupil1.bfpField = maskBest; % phase mask
pupil1.maskDiam_m = 4.2e-3; % phase mask diameter in meters
pupil1.maskDiam_px = 140; % phase mask diameter in pixels
NA=1.4;
f_4f=15e-2; % 4f lens focal length
M=100; % magnification
resizeFactor=1/4; % numerical sampling of EM field (low = better sampling)
gBlur=0.5; % extra PSF blur factor
FOV_r=200;
n1=1.518; % ref index
n2=n1;
nPhotons=1000; % signal photons
bg=2; % background photons per pixel
lambda=670e-9; % wavelength

%% Image generator
figure;
for z=zVec
    x = (rand-0.5)*10e-6; % x position of emitter (random)
    y = (rand-0.5)*10e-6; % y position of emitter (random)
    xyz=[x,y,z]; 
    [img,bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);
    imagesc(img);title(['z = ' num2str(z)]);pause(1);
end


