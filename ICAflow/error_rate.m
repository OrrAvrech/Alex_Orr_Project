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
%             im = (im - min(im(:))) / (max(im(:)) - min(im(:))) + 1;
            im = (im - mean(im(:))) ./ std(im(:));
            im = abs(im);
%             im = histeq(im);
            cd (original_path);
           
            figure(ii); subplot 121 ; imagesc (im);
                        title('Original');
                        subplot 122 ; imagesc (IC(:,:,Ind_vec(ii)));
                        title('ICA Estimation');
            ii = ii + 1;
        end
    end
end

%% Separate ICA results by plane index
% IC_z = cell(1,num_planes);
% IC_z_inds = [];
% for ii = 1:size(IC,3) %put IC results in the z plane they fit most
%     curr_z = fit_to_plane(IC(:,:,ii));
%     if isempty(IC_z{curr_z}) next_row_ind = 1;
%     else next_row_ind = size(IC_z{curr_z}(1,1,:),3)+1; 
%     end
%     IC_z{curr_z}(1:numel(IC(:,1,1)),1:numel(IC(1,:,1)),next_row_ind) = IC(:,:,ii); 
%     IC_z_inds(end+1) = curr_z;
% end

end

