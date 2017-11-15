function [ Error_vec, Ind_vec ] = error_rateRnd(Emitters, IC, Criterion, Display)
%% Get error per emitter and its matching index

NumSourcesEst = size(IC, 3);
Ind_vec = zeros(1, NumSourcesEst);
Error_vec = zeros(1, NumSourcesEst);

% MaxSources = size(Emitters.dat.y, 3);
% Ind_vec = zeros(1, MaxSources);
% Error_vec = zeros(1, MaxSources);

ii = 1;
    for kk = 1 : NumSourcesEst
        im = Emitters.y(:,:,kk);
        [Error, Ind] = match_by_crit(im, IC, Criterion, Ind_vec);
        Ind_vec(ii) = Ind; 
        Error_vec(ii) = Error;
        ii = ii + 1;
    end

%% img display
if Display == 1
    ii = 1;
        for kk = 1 : NumSourcesEst
            im = Emitters.y(:,:,kk);
            % Mean Normalization of Original Sources
%             im = (im - mean(im(:))) ./ std(im(:));
%             im = abs(im);
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