function [ IC ] = Tetrapod_ICA(IC_num,lin_comb)
%% Mixed Sources Matrix
size_lincomb = size(lin_comb) ;
x = zeros(size_lincomb(3), size_lincomb(1)*size_lincomb(2)) ;
for i=1:size_lincomb(3)
    x(i,:) = reshape(lin_comb(:,:,i), 1, []) ;
end

%% Movie
% if exist(movie_flag) == 1 %movie_flag global
%     show_movie = movie_flag;
% else
%     show_movie = 0;
% end
% if show_movie == 1
%     for i=1:size_lincomb(3)
%         imagesc(Sequence2.LinearCombinations(:,:,i));
%         pause (1) ;
%     end
% end
%% FastICA -- to obtain Mixing_mat and original sources
% num_planes = size_lincomb(3);
% num_emitters = IC_num;
% num_sources = num_planes * num_emitters;

original_path = cdir('..\FastICA\');
[A,W] = fastica(x, 'numOfIC', IC_num) ;
cd (original_path);

s = W * x;
s_img_mat = zeros(size_lincomb(1), size_lincomb(2), IC_num);
for j=1 : IC_num
    s_img_mat(:,:,j) = reshape(s(j,:), size_lincomb(1), size_lincomb(2));
%     figure(j);
%     imagesc(s_img_mat(:,:,j));
end
IC = s_img_mat;
end