function [ Error_vec, Ind_vecNZ ] = error_rateRnd(Emitters, IC, Criterion, Display)
%% Get error per emitter and its matching index

NumSourcesEst = size(IC, 3);
Ind_vec = zeros(1, NumSourcesEst);
Error_vec = zeros(1, NumSourcesEst);

ii = 1;
    for kk = 1 : NumSourcesEst
        im = Emitters.labels(:,:,kk);
        % MeanStd Normalization
            im = (im - mean(im(:)))./std(im(:));
        [Error, Ind] = match_by_crit(im, IC, Criterion, Ind_vec); % Both normalized
        Ind_vec(ii) = Ind; 
        Error_vec(ii) = Error;
        ii = ii + 1;
    end

%% img display
Ind_vecNZ = Ind_vec(Ind_vec>0);
NumSourcesEst = length(Ind_vecNZ);
if Display == 1
    ii = 1;
        for kk = 1 : NumSourcesEst
            im = Emitters.labels(:,:,kk);     
            figure(ii); subplot 121 ; imagesc (im);
                        title('Original');
                        subplot 122 ; imagesc (IC(:,:,Ind_vecNZ(ii)));
                        title('ICA Estimation');
            ii = ii + 1;
        end
end

end