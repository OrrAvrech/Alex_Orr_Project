function [imgCube,bfpField] = imgGenerator_fromPupilFunc_new(pupilFunc,gBlur,nomFocusVec,xyz,signalVec,backgroundVec,FOV_r,lambda,n1,n2,NA,f_4f,M,resizeFactor)
x0y0=xyz([1,2]);
z0Vec=xyz(3);
pupil = getfield(pupilFunc,'bfpField');
maskDiam_m = getfield(pupilFunc,'maskDiam_m');
maskDiam_px = getfield(pupilFunc,'maskDiam_px');

%% ESTABLISH COORDINATE SYSTEMS - bfp and CCD plane
x0 = x0y0(1);
y0 = x0y0(2);

% get cropping coordinates
rr = (FOV_r - 1)/2;
cc = rr;

% EMCCD details
CCDpixelSize = 16e-6; % physical CCD pixel size [m]
% M = CCDpixelSize/pixSizeAfterMag; % microscope magnification

% physical size of back focal plane objects: mask and E field
BFPdiam_m = 2*f_4f*NA/sqrt(M^2 - NA^2); % diameter of E field in bfp (region of E field support) [m]

% establish physical coordinates in back focal plane
L = size(pupil,1); % phase mask length (number of pixels on one side) [px]
px_per_m = maskDiam_px/maskDiam_m; % pixels per meter (in back focal plane)
BFPdiam_px = BFPdiam_m*px_per_m; %  % diameter of bfp (region of E field support) [px]
xPhys = ((1:L) - ceil(L/2))/px_per_m;
[XI,ETA] = meshgrid(xPhys,xPhys); % physical coordinates in SLM space [m]

% determine padding value to convert between units
% resizeFactor = 1/5; % 0.3891
padVal = round((lambda*f_4f*px_per_m/(resizeFactor*CCDpixelSize) - L)/2); % convert k -> position

% establish angular coordinates in back focal plane
xAng = linspace(-1,1,L)*NA/n1*L/BFPdiam_px;
[XX,YY] = meshgrid(xAng,xAng); % each pixel is NA/(BFPdiam_px/2*n1) wide
r = sqrt(XX.^2+YY.^2); % radial coordinate s.t. r = NA/n1 at edge of E field support

% calculate angles in bfp for both refractive indices
k1=2*pi*n1/(lambda); % wavevector, immersion medium
sin_theta = r; % collimation of rays in bfp
cos_theta = sqrt(1-sin_theta.^2);
cos_theta(abs(imag(cos_theta))>0) = 0;
k2=2*pi*n2/(lambda); % wavevector, imaging medium
sin_theta2 = n1/n2*sin_theta; % refraction
cos_theta2 = sqrt(1-sin_theta2.^2);
cos_theta2(abs(imag(cos_theta2))>0) = 0;

%% ADD LINEAR PHASE DUE TO x0, y0
linPhase = exp(1i*2*pi*(XI*M*x0 + ETA*M*y0)/(lambda*f_4f));
bfpField = pupil.*linPhase;

%% GENERATE INDIVIDUAL IMAGES ACCORDING TO NOMFOCUS & Z0 VALUES
imgCube = nan([max(FOV_r) max(FOV_r) length(nomFocusVec)]);

for ii = 1:size(imgCube,3)
    nomFocus = nomFocusVec(ii);
    z0 = z0Vec(ii);
    
    % defocus and depth aberrations
    defocusAberr = exp(1i*k1*(-nomFocus)*cos_theta);
    depthAberr = exp(1i*k2*z0*cos_theta2);

    currPupil = defocusAberr.*depthAberr.*bfpField;

    % generate image from pupil function
    Eimg_OS = fftshift(fft2(ifftshift(padarray(currPupil,[padVal padVal],'both')))); % oversampled image plane guess
    Iimg_OS = Eimg_OS.*conj(Eimg_OS); % intensity at image plane (oversampled)
    Iimg_full = imresize(Iimg_OS,resizeFactor,'bilinear'); % downsample to CCD pixel size

    % scale signal, add background
    Iimg = Iimg_full(round(end/2-cc):round(end/2+cc),round(end/2-rr):round(end/2+rr)); % crop image
    Iimg = Iimg*signalVec(ii)/sum(Iimg(:)); % rescale signal
    Iimg = Iimg + max(0,backgroundVec(ii)); % add background

    % add blur
    if gBlur > 0
        h = fspecial('gaussian',[5 5],gBlur);
        Iimg = imfilter(Iimg,h,'replicate');
    end
    
    % store result
    imgCube(:,:,ii) = Iimg;
end

end