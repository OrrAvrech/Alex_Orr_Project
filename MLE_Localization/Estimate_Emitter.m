function thetaOracle = Estimate_Emitter(input_image)

% load the provided 3mu Tetrapode phase mask
% load('Tetra3_angle.mat');
load('SPmask4um.mat');

% setup imaging system parameters
setup_param.NA = 1.4;% objective numerical aperature [a.u.]
setup_param.n = 1.518;% refractive index - assumed to be matched [a.u.]
setup_param.M = 100;% the microscope magnification factor [a.u.]
setup_param.lambda = 670*1e-9;% the emission wavelength [m]
setup_param.f4f = 15*1e-2;% the focal length of each 4f lens [m]
setup_param.N = 1000; % number of photons per psf [photons]
setup_param.b = 2; % constant background noise level [a.u]

% setup phase parameters
phase_param.phase_mask = maskBest; % provided phase mask [complex]
phase_param.padSize = 0; % padding in each side [pixels]
phase_param.rotAngle = 0; % rotation angle [degrees]

% z-range to create images for 
% z_range = (1.5:-0.1:-1.5)*(1e-6); % [m]

% dimensions of the image (assuming square dims)
% Npixels = size(maskBest,1);

% calculate the mask width in pixels
% maskWidth = nnz(maskBest(round(Npixels/2),:));

% calculate camera pixel size [m]
% psize = (maskWidth/Npixels)*setup_param.lambda/(2*setup_param.NA); %pixel size

%% MLE
% extract constraints by oracle xy locations and given precision
% xy_gt = xyz_gt(:,1:2); % [m]
nemitters = 1;
xy_ = [0.1, 0.2]*1e-6; %TODO: decide on right tuning 
precision = 3e-9; % [m] 
[lb,ub] = ConstraintsOracleXY(xy_,precision); %[m] %TODO: decide on right tuning 

% intensity and background noise are lower bounded by zero
lb(3*nemitters+1:end) = zeros(nemitters+1,1);


dim = size(input_image);
PSF = @(r) imcrop(Point_Emitter_PSF(r,setup_param,phase_param),[96,96,dim(1)-1,dim(2)-1]);
% resulting negative loglikelihood function to minimize
costML = @(theta) OracleCostML(input_image,PSF,theta,nemitters,dim);

% starting point - initial guess
x0 = [60, 20, -1, 5, 1]*1e-6; %TODO: decide on right tuning 

% use the interior point method by fmincon to localize
options = optimoptions(@fmincon,'Display','iter');
thetaOracle = fmincon(costML,x0,[],[],[],[],lb,ub,[],options);


end