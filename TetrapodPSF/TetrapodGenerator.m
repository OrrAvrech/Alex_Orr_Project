function [ Sequence ] = TetrapodGenerator( x_cell, y_cell, n, NumFrames )

% Receives emitter locations (x,y), num of z planes and num of frames
% Outputs Tetrapod Sequence


rng(33545);

% NOTE: For this to work, I need to add the local W_i GNGC to the layer separation code.

%% Load masks - needed for the tetrapod PSF generation
load('SPmask4um.mat','maskBest');

%% Set parameters for the tetrapod PSF generation, per each z plane
nomFocusVec        = 0;                        % focal plane (0 = interface)
% n                  = 10;                       % n+1 is the number of z planes
zVec               = linspace(-2e-6,2e-6,n+1); % z positions of emitter
pupil1.bfpField    = maskBest;                 % phase mask
pupil1.maskDiam_m  = 4.2e-3;                   % phase mask diameter in meters
pupil1.maskDiam_px = 140;                      % phase mask diameter in pixels
NA                 = 1.4;                      % Numerical aperture of the microscope
f_4f               = 15e-2;                    % 4f lens focal length
M                  = 100;                      % magnification
resizeFactor       = 1/4;                      % numerical sampling of EM field (low = better sampling)
gBlur              = 0.5;                      % extra PSF blur factor
FOV_r              = 200;                      % Field of view of the image. Measured in #pixels and assumed squared image
n1                 = 1.518;                    % ref index
n2                 = n1;                       % ???
nPhotons           = 1000;                     % signal photons
bg                 = 2;                        % background photons per pixel
lambda             = 670e-9;                   % Imaging wavelength

% %% Image generator
% figure;
% for z=zVec
%     x = (rand-0.5)*10e-6; % x position of emitter (random)
%     y = (rand-0.5)*10e-6; % y position of emitter (random)
%     xyz=[x,y,z]; 
%     [img,bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);
%     imagesc(img);title(['z = ' num2str(z)]);pause(1);
% end

%% Create figures with different z values and combine

%%%%%%%%%%%%%%%%%%%%%%% Function Parameters %%%%%%%%%%%%%%%%%%%%%
% NumOfEmitters     = 3;      % Per layer
% ZposIndex         = [1]; % Planes along z to take from
% NumFrames         = 4;      % Number of frames in the movie

ZposIndex        = 1:n; % Planes along z to take from
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Flags (all binaries)
ApplyBlinkingFlag = 1; % If 1, then each isolated emitter is multiplied by a random number to simulate blinking
AddNoiseFlag      = 0; % Currently not in use

% Minimal number of frames should be 2
if NumFrames < 2
    NumFrames = 2;
end

% Stack initialization
ImPlaneZ          = zeros(FOV_r, FOV_r, length(ZposIndex), NumFrames);

% Noise
if AddNoiseFlag
    FrameNoise = -1;
else
    FrameNoise = zeros(FOV_r, FOV_r);
end

kk = 1;

% For each z layer
for ii = 1:length(ZposIndex)
    % Generate a random number of emitters for each layer
    %%%%%%%%%%%%%%%%%%%%%%% Function Parameters %%%%%%%%%%%%%%%%%%%%%%%
%     x = (rand(NumOfEmitters, 1)-0.5)*10e-6; % x position of emitter (random)
%     y = (rand(NumOfEmitters, 1)-0.5)*10e-6; % y position of emitter (random)
      x = x_cell{ii};
      y = y_cell{ii};
      NumOfEmitters = numel(x_cell{ii});  % Per Layer
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % For each frame in the movie
    for FrameInd = 1:NumFrames
        % Generate blinking pattern
        if ApplyBlinkingFlag
            BlinkingVec = rand(NumOfEmitters, 1);
        else
            BlinkingVec = ones(NumOfEmitters, 1);
        end
        Sequence.BlinkingVec(:, kk) = BlinkingVec;
        
        % Generate each frame: Generate tetrapod PSF for each emitter (per z layer, per frame)
        % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for jj = 1:NumOfEmitters
            xyz = [x(jj), y(jj), zVec(ZposIndex(ii))];
            
            % Generate PSF
            [img, bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,...
                FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);
            
            % Accumulate in a stack
            %%% ------------------------------------------------------------
            ImPlaneZ(:, :, ii, FrameInd) = ImPlaneZ(:, :, ii, FrameInd) + BlinkingVec(jj)*img;
            %%% ------------------------------------------------------------
        end
        % Add noise
        % NEED TO ADD NOISE GENERATION PER FRAME
        ImPlaneZ(:, :, ii)     = ImPlaneZ(:, :, ii) + FrameNoise;
        
        % Normalize each layer for values 0-255
        ImPlaneZ(:, :, ii)     = ImPlaneZ(:, :, ii)/max(max(ImPlaneZ(:, :, ii)))*255;
        % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        % Rearrange in blocks of [x, y, t, z]. This is for presentation convenience only
        eval(['Sequence.Planes.z' num2str(ZposIndex(ii)) ' = squeeze(ImPlaneZ(:, :, ' num2str(ii) ', :));']);
        
        kk = kk + 1;
    end
end

%% Take different linear combinations of the planes
LinComb        = zeros(FOV_r, FOV_r, NumFrames);
Weights_Acccum = zeros(length(ZposIndex), NumFrames);
for ii = 1:NumFrames
    % Generate weights between [0, 1] - this part is legacy
%     Weights = rand(length(ZposIndex), 1);
    Weights = ones(length(ZposIndex), 1);
    
    % Take linear combination
    LinComb(:, :, ii)     = sum(repmat(permute(Weights, [2 3 1]), [FOV_r, FOV_r, 1]).*squeeze(ImPlaneZ(:, :, :, ii)), 3);
    %LinComb(:, :, ii)     = LinComb(:, :, ii)./max(max(LinComb(:, :, ii)))*255;
    
    % Accumulate weights
    Weights_Acccum(:, ii) = Weights;
end

% NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% ImPlaneZ is of dimensions [x, y, z, t] ~
% LinComb  is of dimensions [x, y, t]    ~
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Save output
%% ----------------------------------------------------------------------------------------------------
% Put in structure form
Sequence.LinearCombinations = LinComb;
Sequence.Weights            = Weights_Acccum;
Sequence.ImPlaneZ           = ImPlaneZ;

Sequence2 = Sequence;

% Save as mat
% save Sequence2 Sequence2








end

