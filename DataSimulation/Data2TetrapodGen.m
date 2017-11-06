function [ Sequence2 ] = Data2TetrapodGen( Emitters, NumFrames )

rng(33545);

% NOTE: For this to work, I need to add the local W_i GNGC to the layer separation code.
% NOTE: If Emitters' class is not struct --> Generate Single Tetrapod Image

%% Load masks - needed for the tetrapod PSF generation
original_path = cdir('..\TetrapodPSF\');
load('SPmask4um.mat','maskBest');

%% Set parameters for the tetrapod PSF generation, per each z plane
nomFocusVec        = 0;                        % focal plane (0 = interface)
% n                  = 10;                       % n+1 is the number of z planes
% zVec               = linspace(-2e-6,2e-6,n+1); % z positions of emitter
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

%% Create figures with different z values and combine
% Flags
AddNoiseFlag = 0;
flag_Save2DS = 0; 
% add gaussian noise

if class(Emitters) == 'struct'
%%%%%%%%%%%%%%%%%%%%%%% Function Parameters %%%%%%%%%%%%%%%%%%%%%
    x             = Emitters.x;         % Vec of random x coordinates
    y             = Emitters.y;         % Vec of random y coordinates
    zVec          = Emitters.zVec;      % Possible z planes
    ZposIndex     = Emitters.ZposIndex; % Indices of zVec (elements may be equal)
    NumOfEmitters = numel(ZposIndex);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Flags (all binaries)
    ApplyBlinkingFlag = 1; % If 1, then each isolated emitter is multiplied by a random number to simulate blinking

    % Minimal number of frames should be 2
    if NumFrames < 2
        NumFrames = 2;
    end

    % Stack initialization
    ImPlaneZ          = zeros(FOV_r, FOV_r, length(ZposIndex), NumFrames);

    % For each z layer
    for ii = 1:length(ZposIndex)

        if flag_Save2DS == 1 % Currently not in use            
            labels_mat = [];
            for jj = 1:NumOfEmitters
                xyz = [x(jj), y(jj), zVec(ZposIndex(ii))];

                % Generate PSF
                [img, bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,...
                    FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);

                labels_mat(:,end+1) = reshape(img, [], 1);
            end 
            
        else
            % For each frame in the movie
            for FrameInd = 1:NumFrames
                % Generate blinking pattern
                if ApplyBlinkingFlag
                    BlinkingVec = rand(NumOfEmitters, 1);
                else
                    BlinkingVec = ones(NumOfEmitters, 1);
                end

                % Generate each frame: Generate tetrapod PSF for each emitter (per z layer, per frame)
                % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    xyz = [x(ii), y(ii), zVec(ZposIndex(ii))];

                    % Generate PSF
                    [img, bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,...
                        FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);                  

                    % Noise
                    if AddNoiseFlag == 1
                        img = poissrnd(img);
                    end

                    % Accumulate in a stack
                    %%% ------------------------------------------------------------
                    ImPlaneZ(:, :, ii, FrameInd) = ImPlaneZ(:, :, ii, FrameInd) + BlinkingVec(ii)*img;
                    %%% ------------------------------------------------------------
                % Add noise
                % NEED TO ADD NOISE GENERATION PER FRAME
                ImPlaneZ(:, :, ii)     = ImPlaneZ(:, :, ii);

                % Normalize each layer for values 0-255
                ImPlaneZ(:, :, ii)     = ImPlaneZ(:, :, ii)/max(max(ImPlaneZ(:, :, ii)))*255;
                % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                % Rearrange in blocks of [x, y, t, z]. This is for presentation convenience only
                eval(['Sequence.Planes.z' num2str(ZposIndex(ii)) ' = squeeze(ImPlaneZ(:, :, ' num2str(ii) ', :));']);
                
            end
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
    cd (original_path);
    % Save as mat
    % save Sequence2 Sequence2

    %% Save Tetrapod Data to DataSet
    % DataSet.X: a cell of samples. Each sample is a movie of blinking LinearCombinations
    % DataSet.y: a cell of tags. Each tag is a packet of source images

    features = cell(1);
    labels   = cell(1);
    if flag_Save2DS == 1 % Currently not in use
        features{:} = reshape(Sequence2.LinearCombinations, [], 1);
        labels{:}   = reshape(labels_mat, [], 1);
        m = matfile('DataSet.mat','Writable',true);
        try
            m.features = [m.features, features];
            m.labels   = [m.labels, labels];
        catch
            save('DataSet.mat', 'features', 'labels', '-v7.3');
        end
    end

else
    % Single Tetrapod Generator
    
    xyz = Emitters;
    % Generate PSF
    [img, bfpField] = imgGenerator_fromPupilFunc_new(pupil1,gBlur,nomFocusVec,xyz,nPhotons,bg,...
                    FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor);
                
    % Noise
    if AddNoiseFlag == 1
        img = poissrnd(img);
    end
    
    % Sequence is returned as a Single Tetrapod Image
    Sequence2 = img;
    cd (original_path);
end
    

end



