clc; close all;
%% Set parameters for data generator

NumZplanes  = 20;                                 
NumSamples  = 1000;                               % numer of rand and param. points
zVec        = linspace(-2e-6, 2e-6, NumZplanes+1); % z positions of emitter
t           = linspace(-4e-6, 4e-6, NumSamples);   % scaled according to zVec
xt          = 2e-6*cos((30*pi)/(4e-6)*t);
yt          = 2e-6*sin((30*pi)/(4e-6)*t);
zt          = t;
MatchErr    = 0.5e-6;                              % param. and lottery points error

vslz_ParamAndRand = 1;                             % plot param. and rand points when 1 
vslz_MatchPts     = 1;                             % plot matched points when 1
vslz_Emitters     = 1;                             % plot chosen emitters when 1

flag_Save2File    = 0;                             % append EmittersObj
                                                   % to DataObjects.mat
                                                   % when 1. Save2File at
                                                   % end of process

%% Allow Repetitions
% acquire different values of random xyz in each iteration

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Uncomment the following when used and add end of loop %%%
% NumRep = 50;
% jj = 0;
% while jj < NumRep
%     jj = jj + 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x           = 4e-6*randn(1, NumSamples);           % rand positions should be scaled
y           = 4e-6*randn(1, NumSamples);
z           = 4e-6*randn(1, NumSamples);

%% Plot parametrization and random points

if vslz_ParamAndRand == 1
    
    figure(1);
    plot3(xt,yt,zt);
    hold on;

    scatter3(x,y,z);
    hold off;
    
end

%% Find matched points

c = find(x <= xt + MatchErr & x >= xt - MatchErr & y <= yt + MatchErr...
    & y >= yt - MatchErr & z <= zt + MatchErr & z >= zt - MatchErr);

x_match = xt(c);
y_match = yt(c);
z_match = zt(c);

if (abs(max(x_match))> 10e-6 || abs(min(x_match))< 0.1e-9) ||...
        abs((max(y_match))> 10e-6 || abs(min(y_match))< 0.1e-9)
    sprintf('Invalid emitters locations. Generate again')
    return
end    

%% Plot matched points

if vslz_MatchPts == 1
    
    figure(2);
    plot3(xt,yt,zt);
    hold on;
    scatter3(x_match,y_match,z_match, 'filled');
    hold off;
    
end

%% Separate to z-planes and obtain emitters

% x_cell and y_cell are the emitters locations in each z-plane
% e.g (x_cell{1},y_cell{1}) is an emitter's (x,y) in z=zVec(1)

zVecDiff = abs(zVec(1)-zVec(2));
x_cell = cell(1, numel(zVec));
y_cell = cell(1, numel(zVec));
kk_vec = ones(1, numel(zVec));
for jj = 1 : numel(z_match)
    for ii = 1 : numel(zVec)
        if ((z_match(jj) >= zVec(ii) - zVecDiff/2) && (z_match(jj) <= zVec(ii) + zVecDiff/2))
            x_cell{ii}(kk_vec(ii),:) = x_match(jj);
            y_cell{ii}(kk_vec(ii),:) = y_match(jj);
            kk_vec(ii) = kk_vec(ii) + 1;
        end
    end
end

%% Plot Emitters

if vslz_Emitters == 1
    
    figure(3);
    plot3(xt,yt,zt);
    hold on;

    for ii = 1 : numel(zVec)
        scatter3(x_cell{ii}, y_cell{ii}, zVec(ii)*ones(1,numel(x_cell{ii})), 'filled');
    end
    hold off;
    
end

%% Create Emitters Object and save data to file

EmittersObj.x    = x_cell;
EmittersObj.y    = y_cell;
EmittersObj.zVec = zVec;

if flag_Save2File == 1
    m = matfile('DataObjects.mat','Writable',true);
    try
        m.EmittersObj = [m.EmittersObj, EmittersObj];
    catch
        save('DataObjects.mat', 'EmittersObj', '-v7.3');
    end

    whos('-file','DataObjects.mat')
end

% for ii = 1 : numel(zVec)
%     for jj = 1 : numel(x_cell{ii})
%         xSrc = x_cell{ii}(jj);
%         ySrc = y_cell{ii}(jj);
%         zSrc = zVec(ii);
%         xyz = [xSrc ySrc zSrc];
%     end
% end
