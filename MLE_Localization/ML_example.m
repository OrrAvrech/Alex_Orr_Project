%% Script for Good Guys at Boris' Project

% start with a clean slate
close all; clear all; clc;

% load the provided 3mu Tetrapode phase mask
load('Tetra3_angle.mat');

% setup imaging system parameters
setup_param.NA = 1.45;% objective numerical aperature [a.u.]
setup_param.n = 1.518;% refractive index - assumed to be matched [a.u.]
setup_param.M = 100;% the microscope magnification factor [a.u.]
setup_param.lambda = 670*1e-9;% the emission wavelength [m]
setup_param.f4f = 15*1e-2;% the focal length of each 4f lens [m]
setup_param.N = 2000; % number of photons per psf [photons]
setup_param.b = 2; % constant background noise level [a.u]

% setup phase parameters
phase_param.phase_mask = maskBest; % provided phase mask [complex]
phase_param.padSize = 0; % padding in each side [pixels]
phase_param.rotAngle = 0; % rotation angle [degrees]

% z-range to create images for 
z_range = (1.5:-0.1:-1.5)*(1e-6); % [m]

% dimensions of the image (assuming square dims)
Npixels = size(maskBest,1);

% calculate the mask width in pixels
maskWidth = nnz(maskBest(round(Npixels/2),:));

% calculate camera pixel size [m]
psize = (maskWidth/Npixels)*setup_param.lambda/(2*setup_param.NA); %pixel size

% dimensions of the simulated patch
dim = [64 64];

% cropped PSF generator to reduce complexity
PSF = @(r) imcrop(Point_Emitter_PSF(r,setup_param,phase_param),[96,96,dim(1)-1,dim(2)-1]);

% number of emitters
nemitters = 1;

% signal photons and background
Nph = setup_param.N;
b = setup_param.b;

% create high-resolution grid in [m]
[X,Y] = meshgrid((1:dim(1)) - ceil(dim(1)/2));
xhigh = X*psize;
yhigh = flipud(Y)*psize;

% choose close by points in XY to be the ground truth 
xyz_gt(1,:) = [xhigh(36,32) yhigh(36,32) z_range(1)]; %32,34
% xyz_gt(2,:) = [xhigh(32,34) yhigh(32,34) z_range(16)]; %32,32
% xyz_gt(3,:) = [xhigh(36,37) yhigh(36,37) z_range(end)]; %35,36
% xyz_gt(4,:) = [xhigh(29,30) yhigh(29,30) z_range(3)]; %28,29

% simulate data with X,Y locations instead of convolving spikes
dataML = zeros(dim);
for i=1:nemitters
    dataML = dataML + Nph*PSF([xyz_gt(i,1),xyz_gt(i,2),xyz_gt(i,3)]);
end

% truncate small numerical errors on the order of 1e-16
dataML(dataML<0) = 0;

% add a constant background noise b
% datan = dataML + setup_param.b*max(dataML(:));
datan = dataML + b; % 2 photons background

% maximal value in data
maxVal = max(datan(:));
    
% apply the possion noise statistics to get the measured image
datan = maxVal*double(imnoise(im2uint8(datan/maxVal),'poisson'))/255;

% plot simulated data with psf locations
% figure();imagesc(datan);colormap(hot);axis off;axis square;hold on;
% plot([32 34 37 30]+1,[36 32 36 29]+1,'b+','LineWidth',2);hl = legend('$Centers$');
% ht = title('$Simulated \ 4 \ emitters$');set([hl ht],'interpreter','latex');
figure();imagesc(datan);colormap(hot);axis off;axis square;hold on;
plot(32+1,36+1,'b+','LineWidth',2);hl = legend('$Centers$');
ht = title(['$Simulated \ ', num2str(nemitters), '\ emitters$']);set([hl ht],'interpreter','latex');

% extract constraints by oracle xy locations and given precision
xy_gt = xyz_gt(:,1:2); % [m]
precision = 3e-9; % [m]
[lb,ub] = ConstraintsOracleXY(xy_gt,precision); %[m]

% intensity and background noise are lower bounded by zero
lb(3*nemitters+1:end) = zeros(nemitters+1,1);

% resulting negative loglikelihood function to minimize
costML = @(theta) OracleCostML(datan,PSF,theta,nemitters,dim);

% starting point - initial guess
% x0 = [xy_gt(:,1);xy_gt(:,2);[0.5 0 -0.5 0.5]'*1e-6;ones(nemitters,1);1e-3]; %,-precision/2
x0 = [xy_gt(:,1);xy_gt(:,2);[0.5 0 -0.5 0.5]'*1e-6;Nph*ones(nemitters,1);b];

% use the interior point method by fmincon to localize
options = optimoptions(@fmincon,'Display','iter');
thetaOracle = fmincon(costML,x0,[],[],[],[],lb,ub,[],options);

% plot recovered positions relative to ground truth
figure();stem3(xyz_gt(:,1)*1e6,xyz_gt(:,2)*1e6,xyz_gt(:,3)*1e6,'o');hold on;
stem3(thetaOracle(1:nemitters)*1e6,thetaOracle(nemitters+1:2*nemitters)*1e6,thetaOracle(2*nemitters+1:3*nemitters)*1e6,'+');
hx = xlabel('$x \ [\mu m]$');hy = ylabel('$y \ [\mu m]$');hz = zlabel('$z \ [\mu m]$');
hl = legend('$Ground \ Truth$','$Oracle \ XY$','Location','best');
ht = title('$\hat{z}_{ML} \ with \ XY \ Oracle$');set([hx hy hz  hl ht],'interpreter','latex');

% recovered image using ML estimation
recOracle = zeros(dim);
recEmitters = zeros(dim(1),dim(2),4);
for i=1:nemitters
    recEmitters(:,:,i) = thetaOracle(3*nemitters+i)*PSF([thetaOracle(i),thetaOracle(i+nemitters),thetaOracle(i+2*nemitters)]);
    recOracle = recOracle + recEmitters(:,:,i);
end

% plot recovered patch relative to original
figure();
subplot(1,2,1);imagesc(datan);colormap(hot);axis off;axis square;colorbar;title('Org');
subplot(1,2,2);imagesc(recOracle);colormap(hot);axis off;axis square;colorbar;title('XY Oracle');
figure();
% subplot(2,2,1);
imagesc(recEmitters(:,:,1));colormap(hot);axis off;axis square;title('Source 1');
% subplot(2,2,2);imagesc(recEmitters(:,:,2));colormap(hot);axis off;axis square;title('Source 2');
% subplot(2,2,3);imagesc(recEmitters(:,:,3));colormap(hot);axis off;axis square;title('Source 3');
% subplot(2,2,4);imagesc(recEmitters(:,:,4));colormap(hot);axis off;axis square;title('Source 4');
% suptitle('Recovered Emitters Using XY Oracle');
