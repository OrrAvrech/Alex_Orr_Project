clc; close all;
%% Set parameters for data generator

NumZplanes  = 50;                                 
NumSamples  = 100000;                               % numer of rand and param. points
zVec        = linspace(-2e-6, 2e-6, NumZplanes+1); % z positions of emitter
t           = linspace(-4e-6, 4e-6, NumSamples);   % scaled according to zVec
xt          = (1e-6)*(1+0.25*cos(50*pi*t/(0.5e-6))).*cos(t/(0.5e-6)) ;
yt          = (1e-6)*(1+0.25*cos(50*pi*t/(0.5e-6))).*sin(t/(0.5e-6)) ;
zt          = t+(0.5e-6)*sin(50*pi*t/(0.5e-6));
MatchErr    = 0.5e-6;                              % param. and lottery points error

vslz_ParamAndRand = 0;                             % plot param. and rand points when 1 
vslz_MatchPts     = 1;                             % plot matched points when 1
vslz_Emitters     = 1;                             % plot chosen emitters when 1

flag_Save2File    = 0;                             % append EmittersObj
                                                   % to DataObjects.mat
                                                   % when 1. Save2File at
                                                   % end of process
                                                   
flag_Save2Text    = 0;                             % Saves a desirable param. family 

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

if isempty(z_match)
    sprintf('No emitters. Generate again')
    return
elseif (abs(max(x_match))> 10e-6 || abs(min(x_match))< 0.1e-9) ||...
    abs((max(y_match))> 10e-6 || abs(min(y_match))< 0.1e-9)
    sprintf('Invalid emitters locations. Generate again')
    return
end    

%% Plot matched points

if vslz_MatchPts == 1
    
    figure(2);
    plot3(xt,yt,zt);
    grid on;
    hold on;
    scatter3(x_match,y_match,z_match, 'filled');
    hold off;
    
end

%% Separate to z-planes and obtain emitters

% x_cell and y_cell are the emitters locations in each z-plane
% e.g (x_cell{1},y_cell{1}) is an emitter's (x,y) in z=zVec(1)

NumSources = 0;
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
            NumSources = NumSources + 1;
        end
    end
end

%% Plot Emitters

if vslz_Emitters == 1
    
    figure(3);
    plot3(xt,yt,zt);
    grid on;
    hold on;

    for ii = 1 : numel(zVec)
        scatter3(x_cell{ii}, y_cell{ii}, zVec(ii)*ones(1,numel(x_cell{ii})), 'filled');
    end
    hold off;
    
end

%% Create Emitters Object and save data to file

EmittersObj.x           = x_cell;
EmittersObj.y           = y_cell;
EmittersObj.zVec        = zVec;
EmittersObj.NumSources  = NumSources; 

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

%% Save a parameterization family to text table

if flag_Save2Text == 1
    xt_str      = {'(1e-6)*(1+0.25*cos(50*pi*t/(0.5e-6))).*cos(t/(0.5e-6))'};
    yt_str      = {'(1e-6)*(1+0.25*cos(50*pi*t/(0.5e-6))).*sin(t/(0.5e-6))'};
    zt_str      = {'t+(0.5e-6)*sin(50*pi*t/(0.5e-6))'};
    Tnew = table(xt_str, yt_str, zt_str); 
    T = [T ; Tnew];
    writetable(T, 'ParamTable.txt');
end

