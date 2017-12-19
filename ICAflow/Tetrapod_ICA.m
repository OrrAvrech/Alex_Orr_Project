function [ IC ] = Tetrapod_ICA(MaxSources, lin_comb)
%% Mixed Sources Matrix
size_lincomb = size(lin_comb) ;
x = zeros(size_lincomb(3), size_lincomb(1)*size_lincomb(2)) ;
for i=1:size_lincomb(3)
    x(i,:) = reshape(lin_comb(:,:,i), 1, []) ;
end

%% FastICA -- to obtain Mixing_mat and original sources
original_path = cdir('..\FastICA\');
[A,W] = fastica(x, 'numOfIC', MaxSources) ;
cd (original_path);

s = W * x;
IC_num = size(s, 1);
s_img_mat = zeros(size_lincomb(1), size_lincomb(2), IC_num);
for j = 1 : IC_num
    s_img_mat(:,:,j) = reshape(s(j,:), size_lincomb(1), size_lincomb(2));
    % MeanStd Normalization
    % MinMax Normalization
          SingleEstImg = s_img_mat(:,:,j);
%    SingleEstImg = (SingleEstImg - min(SingleEstImg(:))) / (max(SingleEstImg(:)) - min(SingleEstImg(:)));
         SingleEstImg = SingleEstImg / mean(SingleEstImg(:));
         s_img_mat(:,:,j) = SingleEstImg;
end
IC = s_img_mat;
end