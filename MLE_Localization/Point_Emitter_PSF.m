function [ im_out ] = Point_Emitter_PSF( r, setup_param, phase_param )
% im_out = Point_Emitter_PSF(r,setup_param,phase_param)
% this function calculates the image resulting from a point emitter located
% at r=(x0,y0,z0), with an engineered PSF resulting from a SLM phase mask 
% given by phase_mask. the imaging setup parameters is given in the struct 
% setup_param, and the resulting output image is given in im_out.
% 
% Inputs
% r             -   3x1 vector indicating the spatial position of the
%                   emitter [m]
% setup_param   -   a structure with the imaging system parameters:
%                   NA          - objective numerical aperature [a.u.]
%                   n           - refractive index of the imersion medium 
%                                 (assumed to be matched n~1.518) [a.u.]
%                   f4f         - the focal length of each 4f lens [m]
%                   M           - the microscope magnification factor [a.u.]
%                   lambda      - the emission wavelength [m]
%                   N           - total number of signal photons
%                   b           - constant background photons/pixel    
% phase_param   -   a structure containing the phase parameters
%                   padSize     - the padding to add to the mask in each dir.
%                   phase_mask  - the phase mask as complex image
%                   rotAngle    - the rotation angle to apply [deg.]
%
% Outputs
% im_out        -   the output image
%
% Written by Elias Nehme, 17/07/2017

    % extract system parameters
    NA      = setup_param.NA;
    n       = setup_param.n;
    M       = setup_param.M;
    f4f     = setup_param.f4f;
    lambda  = setup_param.lambda;
%     N       = setup_param.N;
%     b       = setup_param.b;
    
    % extract phase parameters
    phase_mask = phase_param.phase_mask;
    rotAngle   = phase_param.rotAngle;
    padSize    = phase_param.padSize; 
    
    % dimensions of the image (assuming square dims)
    Npixels = size(phase_mask,1);
    
    % calculate the mask width in pixels
    maskWidth = nnz(phase_mask(round(Npixels/2),:));
    
%     % calculate camera pixel size [m]
%     psize = (maskWidth/Npixels)*lambda/NA;

    % point source coordinates
    x0 = r(1);
    y0 = r(2);
    z0 = r(3);
    
%     % limiting radius in fourier plane based on Abbe sine condition [m]      
%     lr_FP = f4f*NA/sqrt(M^2 - NA^2);
    
    % amplitude and phase of phase_mask depending on input format
    if ~isreal(phase_mask)
        phase_amp = abs(phase_mask);
        phase_part = angle(phase_mask);
    else
        phase_amp = abs(phase_mask) > 0;
        phase_part = phase_mask;
    end
    
    % rotate phase part by rotAngle and get final mask
    phase_part_rot = imrotate(phase_part,rotAngle,'bilinear','crop');
    phase_mask = phase_amp.*exp(1i*phase_part_rot);
    
    % pad mask to get higher resolution
    phase_mask = padarray(phase_mask,[padSize padSize],0);
                 
    % create a grid of spatial locations and polar coordinates 
    [X,Y] = meshgrid((1:Npixels) - ceil(Npixels/2));
    [phi,rho] = cart2pol(X*2/maskWidth,flipud(Y)*2/maskWidth); % /lr_FP
    
%     % rescale such that rho=1 at edge of E-field support
%     [~,c] = find(abs(phase_mask));
%     rho = rho./rho(max(c),floor(Npixels/2));
    
    % circular aperature
    circ = rho <= 1;
    
    % electric field in fourier plane due to point source at (0,0,0)
    % assuming an aberration-free system
    E0 = circ./((1-(NA/n*rho).^2).^(0.25)).*abs(phase_mask).*exp(1i*angle(phase_mask));
    
    % phase shift due to lateral displacement (x0,y0)
    phi_lat = 2*pi*NA*M/(lambda*sqrt(M^2-NA^2))*rho.*(x0*cos(phi)+y0*sin(phi));
    
    % under the assumption of index-matched media the phase shift due to
    % axial displacement z0
    phi_ax = 2*pi*n/lambda*z0*sqrt(1-(NA*rho/n).^2);
    
    % final expression for the theoretical electric field in the Fourier plane
    E_x0y0z0 = E0.*exp(1i.*(phi_lat + phi_ax));
    
    % resulting image neglecting the scaling factor
    im_out = abs(fftshift(fft2(ifftshift(E_x0y0z0)))).^2;
    
    % slightly blur resulting image to match a more realistic experimental PSF
    Filter = fspecial('gauss',[10,10],1);
    im_out = imfilter(im_out,Filter);
    
    % normalize the resulting PSF to have sum 1
%     im_out = im_out./max(im_out(:));   
    im_out = im_out./sum(im_out(:));
end

