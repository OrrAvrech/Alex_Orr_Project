function [ Error_vec, Ind_vec ] = error_rate(Emitters, IC, Criterion, Display)
%% Get error per emitter and its matching index

NumPlanes  = length(Emitters.zVec);

Ind_vec = zeros(1, Emitters.NumSources);
Error_vec = zeros(1, Emitters.NumSources);

ii = 1;
for jj = 1 : NumPlanes
    for kk = 1 : numel(Emitters.x{jj})
        xyz = [Emitters.x{jj}(kk), Emitters.y{jj}(kk), Emitters.zVec(jj)];
        original_path = cdir('..\DataSimulation\');
        im  = Data2TetrapodGen(xyz, 0, 0); % Single Tetrapod Image
        cd (original_path);
        [Error, Ind] = match_by_crit(im, IC, Criterion, Ind_vec);
        Ind_vec(ii) = Ind; 
        Error_vec(ii) = Error;
        ii = ii + 1;
    end
end

%% img display
if Display == 1
    ii = 1;
    for jj = 1 : NumPlanes
        for kk = 1 : numel(Emitters.x{jj})
            xyz = [Emitters.x{jj}(kk), Emitters.y{jj}(kk), Emitters.zVec(jj)];
            original_path = cdir('..\DataSimulation\');
            im  = Data2TetrapodGen(xyz, 0, 0); % Single Tetrapod Image
            % Mean Normalization of Original Sources
            im = (im - mean(im(:))) ./ std(im(:));
            im = abs(im);
            cd (original_path);
            % MinMax Normalization of Original Sources
            % im = (im - min(im(:))) / (max(im(:)) - min(im(:))) + 1;
           
            figure(ii); subplot 121 ; imagesc (im);
                        title('Original');
                        subplot 122 ; imagesc (IC(:,:,Ind_vec(ii)));
                        title('ICA Estimation');
            ii = ii + 1;
        end
    end
end

end

