clc;
clear;
close all;

rng(33545);

% This is my original code

%% Load masks
load('SPmask4um.mat','maskBest');

%% Set parameters
nomFocusVec        = 0; % focal plane (0 = interface)
n                  = 10;
zVec               = linspace(-2e-6,2e-6,n+1); % z positions of emitter
pupil1.bfpField    = maskBest; % phase mask
pupil1.maskDiam_m  = 4.2e-3; % phase mask diameter in meters
pupil1.maskDiam_px = 140; % phase mask diameter in pixels
NA                 = 1.4;
f_4f               = 15e-2; % 4f lens focal length
M                  = 100; % magnification
resizeFactor       = 1/4; % numerical sampling of EM field (low = better sampling)
gBlur              = 0.5; % extra PSF blur factor
FOV_r              = 200;
n1                 = 1.518; % ref index
n2                 = n1;
nPhotons           = 1000; % signal photons
bg                 = 2; % background photons per pixel
lambda             = 670e-9; % wavelength

% %% Image generator
% figure;
% for z=zVec
%     x = (rand-0.5)*10e-6; % x position of emitter (random)
%     y = (rand-0.5)*10e-6; % y position of emitter (random)
%     xyz=[x,y,z];
%     [img,bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);
%     imagesc(img);title(['z = ' num2str(z)]);pause(1);
% end

%% Create two figures with different z values and combine
NumOfEmitters     = 6;      % Per layer
ZposIndex         = [1 11]; % Planes along z to take from
NumOfComb         = 2;      % Number of linear combinations
NumFrames         = 2;      % Number of frames in the movie

% Flags (all binaries)
ApplyBlinkingFlag = 1; % If 1, then each isolated emitter is multiplied by a random number to simulate blinking
AddNoiseFlag      = 0; % Currently not in use

% Stack initialization
ImPlaneZ          = zeros(FOV_r, FOV_r, length(ZposIndex));

% Minimal number of frames should be 2
if NumFrames < 2
    NumFrames = 2;
end

% Noise
if AddNoiseFlag
    FrameNoise = -1;
else
    FrameNoise = zeros(FOV_r, FOV_r);
end

% For each z layer
for ii = 1:length(ZposIndex)
    % Generate a random number of emitters for each layer
    x = (rand(NumOfEmitters, 1)-0.5)*10e-6; % x position of emitter (random)
    y = (rand(NumOfEmitters, 1)-0.5)*10e-6; % y position of emitter (random)
    
    % Generate blinking pattern
    if ApplyBlinkingFlag
        BlinkingVec = rand(NumOfEmitters, 1);
    else
        BlinkingVec = ones(NumOfEmitters, 1);
    end
    Sequence.BlinkingVec(:, ii) = BlinkingVec;
    
    % Generate each frame: Generate tetrapod PSF for each emitter (per z layer, per frame)
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for jj = 1:NumOfEmitters
        xyz = [x(jj), y(jj), zVec(ZposIndex(ii))];
        
        % Generate PSF
        [img, bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,...
                                                         FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);
        
        % Accumulate in a stack
        %%% ------------------------------------------------------------
        ImPlaneZ(:, :, ii) = ImPlaneZ(:, :, ii) + BlinkingVec(jj)*img;
        %%% ------------------------------------------------------------
    end
    % Add noise
    % NEED TO ADD NOISE GENERATION PER FRAME
    ImPlaneZ(:, :, ii)     = ImPlaneZ(:, :, ii) + FrameNoise;
    
    % Normalize each layer for values 0-255
    ImPlaneZ(:, :, ii)     = ImPlaneZ(:, :, ii)/max(max(ImPlaneZ(:, :, ii)))*255;
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
end

% Take different linear combinations of the planes
LinComb        = zeros(FOV_r, FOV_r, NumOfComb);
Weights_Acccum = zeros(length(ZposIndex), NumOfComb);
for ii = 1:NumOfComb
    % Generate weights between [0, 1]
%     Weights = rand(length(ZposIndex), 1);
    Weights = ones(length(ZposIndex), 1);    
    
    % Take linear combination
    LinComb(:, :, ii)     = sum(repmat(permute(Weights, [2 3 1]), [FOV_r, FOV_r, 1]).*ImPlaneZ, 3);
    %LinComb(:, :, ii)     = LinComb(:, :, ii)./max(max(LinComb(:, :, ii)))*255;
    
    % Accumulate weights
    Weights_Acccum(:, ii) = Weights;
end

% Save output
%% ----------------------------------------------------------------------------------------------------
% Put in structure form
Sequence.LinearCombinations = LinComb;
Sequence.Weights            = Weights_Acccum;
Sequence.ImPlaneZ           = ImPlaneZ;

% Save as mat
save Sequence Sequence





