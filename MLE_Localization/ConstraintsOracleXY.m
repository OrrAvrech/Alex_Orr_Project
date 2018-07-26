function [lb,ub] = ConstraintsOracleXY(xy_gt,precision)
% [lb,ub] = ConstraintsOracleXY(xy_gt,precision)
% function calculates the lower bound and upper bound constraints on the
% resulting oracle assisted ML cost function parameters.
%
% Inputs
% xy_gt       - the approximate ground truth xy coorinates [m]
% precision   - the ground truth precision [m]
%
% Outputs
% lb          - the resulting lower bounds [m]
% ub          - the resulting upper bounds [m]
%
% Written by Elias Nehme, 03/09/2017

    % number of emitters
    nemitters = size(xy_gt,1);
    
    % resulting lower bound constraints from ground truth    
    lb = -Inf*ones(4*nemitters+1,1);
    lb(1:nemitters) = xy_gt(:,1) - precision;
    lb(nemitters+1:2*nemitters) = xy_gt(:,2) - precision;
        
    % upper bound constraints on x0,y0
    ub = Inf*ones(4*nemitters+1,1);
    ub(1:nemitters) = xy_gt(:,1) + precision;
    ub(nemitters+1:2*nemitters) = xy_gt(:,2) + precision;    

end

