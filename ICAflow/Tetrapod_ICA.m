function [ IC ] = Tetrapod_ICA(MaxSources, lin_comb)
%% Mixed Sources Matrix
size_lincomb = size(lin_comb) ;
x = zeros(size_lincomb(3), size_lincomb(1)*size_lincomb(2)) ;
for i=1:size_lincomb(3)
    x(i,:) = reshape(lin_comb(:,:,i), 1, []) ;
end

%% FastICA -- to obtain Mixing_mat and original sources
% num_planes = size_lincomb(3);
% num_emitters = IC_num;
% num_sources = num_planes * num_emitters;

original_path = cdir('..\FastICA\');
[A,W] = fastica(x, 'numOfIC', MaxSources) ;
cd (original_path);

s = W * x;
IC_num = size(s, 1);
s_img_mat = zeros(size_lincomb(1), size_lincomb(2), IC_num);
for j = 1 : IC_num
    s_img_mat(:,:,j) = reshape(s(j,:), size_lincomb(1), size_lincomb(2));
    % Mean Normalization of Estimated Sources
        SingleEstImg = s_img_mat(:,:,j);
        SingleEstImg = (SingleEstImg - mean(SingleEstImg(:))) ./ std(SingleEstImg(:));
        s_img_mat(:,:,j) = abs(SingleEstImg);
    % MinMax Normalization of Original Sources
%         SingleEstImg = (SingleEstImg - min(SingleEstImg(:))) / (max(SingleEstImg(:)) - min(SingleEstImg(:))) + 1;
%         s_img_mat(:,:,j) = abs(SingleEstImg);
% uint8 -- 0-255
end
IC = s_img_mat;
end