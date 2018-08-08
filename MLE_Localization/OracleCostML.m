function funval = OracleCostML(data,psfmdl,theta0,nemitters,dim,df)
% funval = OracleCostML(data,psfmdl,theta,nemitters)
% function calculates the likelihood of the emitters locations and 
% intensities given the observed data and the psf theoretical model. 
%
% Inputs
% data          -   observed measurement data
% psfmdl        -   the psf theoretical model as function of r
% theta         -   parameters to be estimated (x,y,z[m], N[counts], b[counts] concatenated)
% nemitters     -   the number of emitters in the data [a.u.]
% dim           -   dimension of the high-res image
% df            -   downsampling factor
%               
% Outputs
% fval          -   the resulting negative log-likelihood function value
%
% Written by Elias Nehme, 03/09/2017

         
    % define the parameters to be optimized
    x0 = theta0(1:nemitters);
    y0 = theta0(nemitters+1:2*nemitters);
    z0 = theta0(2*nemitters+1:3*nemitters);
    N = theta0(3*nemitters+1:4*nemitters);
    b = theta0(end);
    
    % calculate the model image
    im_model = zeros(dim);
    for i=1:nemitters
        im_model = im_model + N(i)*psfmdl([x0(i),y0(i),z0(i)]);
    end
    
%     % downsampling to get the theoretical measurement before noise
%     im_model = imresize(im_model,1/df,'box');
    
    % add the background constant noise
%     im_model = im_model + b*max(im_model(:));
    im_model = im_model + b;
    
    % calculate the negative log likelihood of the measurements under the model
    funval = -(sum(data(:).*log(im_model(:))) - sum(im_model(:)));  
end

